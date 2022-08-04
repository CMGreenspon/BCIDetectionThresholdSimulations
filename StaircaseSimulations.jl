using LsqFit, Distributions, GLM, DataFrames

"""
TransformedStaircaseSimulation(ValidStims::Vector{Int},  pDetected::Vector{Float64}; MaxTrials::Int = 1000, 
    NumPermutations::Int = 1000, NumAFC::Int = 2, Criterion::Vector{Int} = [3,1], InitAmp::Int = 0,
    InitStepSize::Int = 20, DecreaseStepCoeff::Float64 = 0.75, IncreaseStepCoeff::Float64 = 0.75,
    MinStepSize::Int = 2, MaxStepSize::Int = 20, MaxReversions::Int = 7, SkipFirstNReversions::Int = 0,
    UseMLE::Bool = false)

A function that simulates runs of a transformed staircase. Several parameters may be given such as
    the reversion criteria (i.e. a 3/1 up/down staircase versus), the stop criteria (number of reversions
    or number of trials), the convergence rate, etc.

returns amplitude_history, detection_history, reversion_history, estimated_thresholds, num_trials
"""
function TransformedStaircaseSimulation(ValidStims::Vector{Int},  pDetected::Vector{Float64}; MaxTrials::Int = 1000, 
    NumPermutations::Int = 1000, NumAFC::Int = 2, Criterion::Vector{Int} = [3,1], InitAmp::Int = 0,
    InitStepSize::Int = 20, DecreaseStepCoeff::Float64 = 0.75, IncreaseStepCoeff::Float64 = 0.75,
    MinStepSize::Int = 2, MaxStepSize::Int = 20, MaxReversions::Int = 7, SkipFirstNReversions::Int = 0,
    UseMLE::Bool = false)

    # Error checking
    if SkipFirstNReversions >= MaxReversions
        error("SkipFirstNReversions >= MaxReversions")
    end

    if UseMLE
        fm = @formula(dt ~ amp)
        if MaxReversions < 4
            error("If UsingMLE -> MaxReversions must be equal to or greater than 4")
        end

        if SkipFirstNReversions > 0
            println("Warning: SkipFirstNReversions is ignored when UsingMLE is True")
        end
    end
    
    # Small computations
    afc_chance = 1/NumAFC
    num_valid_stims = length(ValidStims) 
    max_stim_level = maximum(ValidStims)
    min_stim_level = minimum(ValidStims)

    # Initialize variables for each permutation
    detection_history = fill(NaN, MaxTrials, NumPermutations)
    amplitude_history = fill(NaN, MaxTrials, NumPermutations)
    reversion_history = fill(NaN, MaxTrials, NumPermutations)
    estimated_thresholds = fill(NaN, NumPermutations)
    num_trials = fill(NaN, NumPermutations)

    for p = 1:NumPermutations
        # Initialize variables for trials
        step_size = InitStepSize
        current_direction = 0
        new_direction = 0
        consecutive_answers = [0,0]
        criterion_reached = false
        threshold_found = false
        t = 0
        num_reversions = 0

        if InitAmp == 0 # Choose a random stim
            StimAmp = ValidStims[rand(1:num_valid_stims)]
        else
            StimAmp = InitAmp
        end

        while t <= MaxTrials && !threshold_found
            # Iterate the trial counter
            t += 1
            # Get current stim level
            stim_idx = findall(ValidStims .== StimAmp)
            if length(stim_idx) == 0 # If invalid stim then find nearest valid
                (_, stim_idx) = findmin(abs.(ValidStims .- StimAmp))
                StimAmp = ValidStims[stim_idx[1]]
            end
            amplitude_history[t,p] = StimAmp
            pd_at_stim = pDetected[stim_idx[1]] # Will error if the StimAmp is not a member of ValidStims
            
            # Determine if detected
            if rand() < pd_at_stim || (NumAFC > 1 && rand() < afc_chance)
                # Deterimine if the 'correct' interval was selected. Equal to p(detected) + 1/NumAFC
                detection_history[t,p] = 1
                consecutive_answers[1] += 1
            else
                detection_history[t,p] = 0
                consecutive_answers[2] += 1
            end

            # Evaluate criterion and change StimAmp
            if consecutive_answers[1] >= Criterion[1]
                # Criterion in a row correct means decrease amplitude
                new_direction = -1
                StimAmp = StimAmp - step_size
                consecutive_answers = [0,0]
                criterion_reached = true

            elseif consecutive_answers[2] >= Criterion[2]
                # Criterion in a row incorrect means increase amplitude
                new_direction = 1
                StimAmp = StimAmp + step_size
                consecutive_answers = [0,0]
                criterion_reached = true
            end

            # Ensure StimAmp is in range and is valid (also count as reversion/early stopping criteria)
            if StimAmp > max_stim_level
                StimAmp = max_stim_level
                reversion_history[t,p] = new_direction*-1
            elseif StimAmp < min_stim_level
                StimAmp = min_stim_level
                reversion_history[t,p] = new_direction*-1
            end

            # Update step_size (but only if we've changed stim level at least once)
            if criterion_reached && (new_direction != current_direction)
                if new_direction == 1
                    step_size = step_size * IncreaseStepCoeff
                elseif new_direction == -1
                    step_size = step_size * DecreaseStepCoeff
                end
                
                # Ensure step size is not out of bounds
                if step_size < MinStepSize
                    step_size = MinStepSize
                elseif step_size > MaxStepSize
                    step_size = MaxStepSize
                end

                # Reset boolean
                criterion_reached = false
                # Update direction
                current_direction = new_direction
                # Store reversion index
                reversion_history[t,p] = new_direction*-1
            end

            if ~isnan(reversion_history[t,p]) # Update reversion counter
                num_reversions += 1
            end

            # Stopping criteria
            if num_reversions >= MaxReversions || t == MaxTrials
                if UseMLE # Use Maximum Likelihood Estimation
                    estimated_thresholds[p], _ = ThresholdMLE(reversion_history[:,p], amplitude_history[:,p])
                else # Just average the amplitudes at reversions
                    rev_idx = findall(isnan.(reversion_history[:,p]) .== 0)[1+SkipFirstNReversions:num_reversions]
                    rev_amps = amplitude_history[rev_idx,p]
                    estimated_thresholds[p] = mean(rev_amps)
                end
                num_trials[p] = t # Keep track of the trial number we stopped at  
                threshold_found = true # End while loop
            end
        end
    end

    return amplitude_history, detection_history, reversion_history, estimated_thresholds, num_trials
end


"""
GetTransformedStaircaseTarget(NumAFC::Int, Criterion::Vector{Int})

Provides the convergence point on a psychometric function for a transformed staircase for a given
    criterion and number of alternate forced choices. This is essential for accurately computing
    the error for a staircase as the convergence point is rarely the declared 50% detection threshold.

returns Target
"""
function GetTransformedStaircaseTarget(NumAFC::Int, Criterion::Vector{Int})
    Target = Float64[]
    if NumAFC > 1 # Need to adjust for chance
        chance = 1/NumAFC
        if Criterion[1] >= Criterion[2] && Criterion[2] == 1 
            NR = 0.5^(1/(Criterion[1]/Criterion[2])) # N-th root
            Target = (NR - chance) / (1-chance)
        else
            error("Criterion[1] must be greater than Criterion[2]")
        end
    else
        Target = Criterion[1]/sum(Criterion)
    end
    
    return Target
end


"""
PosthocEstimateStaircaseThreshold(TrialAmplitudes::Matrix{Float64}, ReversionIndices::Matrix{Float64};
    MaxReversions::Int = 7, SkipFirstNReversions::Int = 0, UseMLE::Bool=false)

A function that allows one to determine the estimated threshold by controlling the stop criteria after the fact.
    This requires that the input staircase has a greater stop criteria than passed to this function, e.g.
    if the staircase stopped at 5 reversions you cannot post-hoc see what it would have been if it stopped at 7.

    returns EstimatedThreshold
"""
function PosthocEstimateStaircaseThreshold(TrialAmplitudes::Matrix{Float64}, ReversionIndices::Matrix{Float64};
     MaxReversions::Int = 7, SkipFirstNReversions::Int = 0, UseMLE::Bool=false)

    # Error checking
    if SkipFirstNReversions < 0
         error("SkipFirstNReversions < 0")
    end
    if MaxReversions <= SkipFirstNReversions
        error("MaxReversions <= SkipFirstNReversions")
    end
    if size(ReversionIndices) != size(TrialAmplitudes)
        error("size(ReversionIndices) != size(TrialAmplitudes)")
    end
    if UseMLE && MaxReversions < 4
        error("If UsingMLE -> MaxReversions must be equal to or greater than 4")
    end

    NumPermutations = size(TrialAmplitudes,2)
    # Initalize output
    EstimatedThreshold = fill(NaN, NumPermutations)
    if UseMLE
        EstimatedSigma = zeros(NumPermutations)
        for p = 1:NumPermutations
            EstimatedThreshold[p], EstimatedSigma[p] = ThresholdMLE(ReversionIndices[:,p], TrialAmplitudes[:,p])
        end
        return EstimatedThreshold, EstimatedSigma
    else # Default average reversion indices
        for p = 1:NumPermutations
            # Get the indices at which there was a reversion
            rev_idx = findall(isnan.(ReversionIndices[:,p]) .== 0)[1+SkipFirstNReversions:MaxReversions]
            rev_amps = TrialAmplitudes[rev_idx,p] # Find the amplitude at each reversion
            EstimatedThreshold[p] = mean(rev_amps) # Average the reversion amplitudes
        end
        return EstimatedThreshold
    end

end

function SortedStaircaseStatistics(ThresholdEstimate::Matrix{Float64}, StaircaseStopPoint::Matrix{Float64},
     ThresholdGroundTruth::Float64)

    # First vectorize inputs
    ThresholdEstimate = vec(ThresholdEstimate)
    StaircaseStopPoint = vec(StaircaseStopPoint)
    # Then remove NaN values
    nan_idx = findall(isnan.(ThresholdEstimate) .== 0)
    ThresholdEstimate = ThresholdEstimate[nan_idx]
    StaircaseStopPoint = Int.(StaircaseStopPoint[nan_idx])

    # Find the minimum and maximum number of reversion
    min_repeats = nanminimum(StaircaseStopPoint)
    max_repeats = nanmaximum(StaircaseStopPoint)

    # Initalize outputs
    NumRepeats = min_repeats:max_repeats
    SortedMean = fill(NaN, length(NumRepeats))
    SortedSTD = fill(NaN, length(NumRepeats))
    SortedError = fill(NaN, length(NumRepeats))

    # Find all values matching the number of repeats and perform stats on them
    for (ri, r) in enumerate(NumRepeats)
        r_idx = findall(StaircaseStopPoint .== r)
        SortedMean[ri] = mean(ThresholdEstimate[r_idx])
        SortedSTD[ri] = std(ThresholdEstimate[r_idx])
        SortedError[ri] = mean(abs.(ThresholdEstimate[r_idx].- ThresholdGroundTruth))
    end

    return NumRepeats, SortedMean, SortedSTD, SortedError
end


"""
ThresholdMLE(ReversionIndices::Vector{Float64}, AmplitudeHistory::Vector{Float64})

Subfunction for running MLE on reversion values from staircase. Will get ~NaN valued 
    indices from ReversionIndices and fit the a logistic function to the AmplitudeHistory
    values of those indices. Will treat ReversionIndices == 1 as detected class and 
    ReversionIndices < 0 as non-detected class. Any values besides 1 and -1 may produce errors.

    returns estimated_threshold, estimated_sigma
"""
function ThresholdMLE(ReversionIndices::Vector{Float64}, AmplitudeHistory::Vector{Float64})
    estimated_threshold = Float64[]
    estimated_sigma = Float64[]
    rev_idx = findall(isnan.(ReversionIndices) .== 0)
    rev_amps = AmplitudeHistory[rev_idx]
    rev_dts = Int.(round.(ReversionIndices[rev_idx]))
    # Assert that negative reversions = 0 for logistic regression
    rev_dts[rev_dts .< 0] .= 0
    # Perform logistic regression
    temp_df = DataFrame(amp = rev_amps, dt = rev_dts) 
    try
        fm = @formula(dt ~ amp)
        logit = glm(fm, temp_df, Binomial(), LogitLink())
        fit_coeffs = coef(logit)
        # Convert and save coeffs
        estimated_threshold = abs(fit_coeffs[1])/fit_coeffs[2]
        estimated_sigma = 1.7/fit_coeffs[2]
    catch # Return NaNs
        estimated_threshold = NaN 
        estimated_sigma = NaN
    end

    return estimated_threshold, estimated_sigma
end