using LsqFit, GLM, DataFrames, Optim

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
function TransformedStaircaseSimulation(ValidStims::Vector{Int},
                                        pDetected::Vector{Float64};
                                        MaxTrials::Int = 1000,
                                        NumPermutations::Int = 1000,
                                        NumAFC::Int = 2,
                                        Criterion::Vector{Int} = [3,1],
                                        InitAmp::Int = 0,
                                        InitStepSize::Int = 20,
                                        DecreaseStepCoeff::Float64 = 0.75,
                                        IncreaseStepCoeff::Float64 = 0.75,
                                        MinStepSize::Int = 2,
                                        MaxStepSize::Int = 20,
                                        MaxReversions::Int = 7,
                                        SkipFirstNReversions::Int = 0,
                                        UseMLE::Bool = false)

    # Error checking
    if SkipFirstNReversions >= MaxReversions
        error("SkipFirstNReversions >= MaxReversions")
    end

    if UseMLE
        if MaxReversions < 4
            error("If UsingMLE -> MaxReversions must be equal to or greater than 4")
        end
        if SkipFirstNReversions > 0
            println("Warning: SkipFirstNReversions is ignored when UsingMLE is True")
        end
    end
    
    # Small computations
    afc_chance = 1/NumAFC
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
            StimAmp = rand(ValidStims,1)[1]
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
                criterion_reached = true

            elseif consecutive_answers[2] >= Criterion[2]
                # Criterion in a row incorrect means increase amplitude
                new_direction = 1
                criterion_reached = true
            end

            # Update step_size (but only if we've changed stim level at least once)
            if criterion_reached
                if current_direction != 0 && new_direction != current_direction
                    # Adjust the step size
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

                    # Store reversion index
                    reversion_history[t,p] = new_direction*-1
                end

                # Apply the step
                if new_direction == 1
                    StimAmp += step_size
                elseif new_direction == -1
                    StimAmp -= step_size
                end

                # Ensure StimAmp is in range and is valid (also count as reversion/early stopping criteria)
                if StimAmp > max_stim_level
                    StimAmp = max_stim_level
                    reversion_history[t,p] = new_direction*-1
                elseif StimAmp < min_stim_level
                    StimAmp = min_stim_level
                    reversion_history[t,p] = new_direction*-1
                end

                # Reset criterion & update direction
                criterion_reached = false
                consecutive_answers = [0,0]
                current_direction = new_direction
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
PESTStaircaseSimulation - Runs a staircase using PEST update rules to find the amplitude at which a target performance level is acheived
"""
function PESTStaircaseSimulation(ValidStims::Vector{Int},
                                 pDetected::Vector{Float64};
                                 MaxTrials::Int = 1000,
                                 NumPermutations::Int = 1000,
                                 NumAFC::Int = 2,
                                 TargetPerformance::Float64 = 0.75,
                                 WaldFactor::Int = 1,
                                 WaldTrials::Int = 4,
                                 InitAmp::Int = 0,
                                 InitStepSize::Int = 20,
                                 MinStepSize::Int = 2,
                                 MaxStepSize::Int = 20,
                                 MaxReversions::Int = 7,
                                 UseMLE::Bool = false)

    if UseMLE
        if MaxReversions < 4
            error("If UsingMLE -> MaxReversions must be equal to or greater than 4")
        end
    end

    # Small computations
    afc_chance = 1/NumAFC
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
        threshold_found = false
        num_reversions = 0
        last_step_was_double = false
        sequential_dir_count = 0
        temp_detection_tracker = Int[]
        t = 0 # Trial index

        if InitAmp == 0 # Choose a random stim
            StimAmp = rand(ValidStims,1)[1]
        else
            StimAmp = InitAmp
        end

        # Actual trial logic
        while t <= MaxTrials && !threshold_found
            # Iterate the trial counter
            t += 1
            stim_idx = findall(ValidStims .== StimAmp)
            if length(stim_idx) == 0 # If invalid stim then find nearest valid
                (_, stim_idx) = findmin(abs.(ValidStims .- StimAmp))
                StimAmp = ValidStims[stim_idx[1]]
            end
            amplitude_history[t,p] = StimAmp
            pd_at_stim = pDetected[stim_idx[1]] # Will error if the StimAmp is not a member of ValidStims
            
            # Deterimine if the 'correct' interval was selected. Equal to p(detected) + 1/NumAFC
            if rand() < pd_at_stim || (num_afc > 1 && rand() < afc_chance)
                detection_history[t,p] = 1
                push!(temp_detection_tracker, 1)
            else
                detection_history[t,p] = 0
                push!(temp_detection_tracker, 0)
            end

            # Evaluate Wald-adjusted likelihood, udate stim if criteria is met, reset detection tracker
            if length(temp_detection_tracker) >= WaldTrials
                observed_detections = sum(temp_detection_tracker)
                expected_detections = length(temp_detection_tracker) * TargetPerformance
                wald_bounds = (expected_detections + WaldFactor, expected_detections - WaldFactor)
                if observed_detections > wald_bounds[1] # If more than expected then decrease amp
                    new_direction = -1
                    StimAmp -= step_size
                    reversion_history[t,p] = new_direction*-1
                    temp_detection_tracker = Int[]
                elseif observed_detections < wald_bounds[2]# If less than expected then increase amp
                    new_direction = 1
                    StimAmp += step_size
                    reversion_history[t,p] = new_direction*-1
                    temp_detection_tracker = Int[]
                end
            end

            # Ensure StimAmp is in range and is valid (also count as reversion/early stopping criteria)
            if StimAmp > max_stim_level
                StimAmp = max_stim_level
                reversion_history[t,p] = new_direction*-1
            elseif StimAmp < min_stim_level
                StimAmp = min_stim_level
                reversion_history[t,p] = new_direction*-1
            end

            if ~isnan(reversion_history[t,p]) # Update reversion counter
                num_reversions += 1
            end

            # Determine new update rule - PEST Rules
            if length(temp_detection_tracker) == 0
                if new_direction != current_direction
                    current_direction = new_direction
                    sequential_dir_count = 0
                else
                    sequential_dir_count += 1
                end
                
                # Determine step size
                if sequential_dir_count == 0
                    step_size = step_size / 2 # Half step size on direction change
                    last_step_was_double = false
                elseif sequential_dir_count == 1
                    last_step_was_double = false
                elseif sequential_dir_count == 2 && last_step_was_double
                    step_size = step_size * 2; # Only double if the last reversion was from a double
                    last_step_was_double = true
                elseif sequential_dir_count == 2 && ~last_step_was_double
                    last_step_was_double = false
                elseif sequential_dir_count > 2
                    step_size = step_size * 2 # Double away
                    last_step_was_double = true
                end
        
                # Ensure step size is in range
                if step_size < MinStepSize
                    step_size = MinStepSize
                elseif step_size > MaxStepSize
                    step_size = MaxStepSize
                end
            end

            # Stopping criteria
            if num_reversions >= MaxReversions || t == MaxTrials
                if UseMLE # Use Maximum Likelihood Estimation
                    estimated_thresholds[p], _ = ThresholdMLE(reversion_history[:,p], amplitude_history[:,p])
                else # Take the final stimulus amplitude
                    estimated_thresholds[p] = StimAmp
                end
                num_trials[p] = t # Keep track of the trial number we stopped at  
                threshold_found = true # End while loop
            end
        end
    end

    return amplitude_history, detection_history, reversion_history, estimated_thresholds, num_trials
end

"""
GetPESTStaircaseTarget(NumAFC::Int, Criterion::Vector{Int}) - UNFINISHED

returns Target
"""
function GetPESTStaircaseTarget(PsychometricCurve::UnivariateDistribution, TargetPerformance::Float64, NumAFC::Int)
    Target = Float64[]
    if NumAFC > 1 # Adjust target for NumAFC
        chance = 1/NumAFC
        adjusted_target = (TargetPerformance - chance) / (1-chance)
        Target = quantile(PsychometricCurve, adjusted_target)
    else
        Target = quantile(PsychometricCurve, TargetPerformance)
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
            mr_idx = findall(isnan.(reversion_history[:,p]) .== 0)[MaxReversions]
            EstimatedThreshold[p], EstimatedSigma[p] = ThresholdMLE(ReversionIndices[1:mr_idx,p], TrialAmplitudes[1:mr_idx,p])
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

"""
SortedStaircaseStatistics(ThresholdEstimate::Matrix{Float64}, StaircaseStopPoint::Matrix{Float64},
     ThresholdGroundTruth::Float64)
Post-hoc sorting of stairsase estimates based on number of trails. Takes the predicted threshold, the trial
    number the prediction was made at, and the true threshold.

returns NumRepeats (x), SortedMean, SortedSTD, SortedError
"""
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


function QUESTStaircase(ValidStims::Vector{Int},
                        pDetected::Vector{Float64};
                        MaxTrials::Int = 1000,
                        NumPermutations::Int = 1000,
                        NumAFC::Int = 2,
                        InitAmp::Int = 0,
                        StepSize::Int = 10,
                        NoiseFactor::Number = 10,
                        UpperBounds::Vector{Float64} = [1.0, 100],
                        LowerBounds::Vector{Float64} = [0.0, -100],
                        InitialGuess::Vector{Float64} = [0.1, 0.0])
    
    # Small computations
    afc_chance = 1/NumAFC
    max_stim_level = maximum(ValidStims)
    min_stim_level = minimum(ValidStims)

    # Initialize variables for each permutation
    detection_history = fill(false, MaxTrials, NumPermutations)
    amplitude_history = fill(0, MaxTrials, NumPermutations)
    estimated_threshold = fill(NaN, MaxTrials, NumPermutations)

    for p = 1:NumPermutations
        # Initialize variables for trials
        t = 0
        if InitAmp == 0 # Choose a random stim
            StimAmp = rand(ValidStims,1)[1]
        else
            StimAmp = InitAmp
        end
        PermGuess = InitialGuess

        # Begin the staircase
        while t < MaxTrials
            # Iterate the trial counter
            t += 1
            amplitude_history[t, p] = StimAmp
            stim_idx = findall(valid_stims .== StimAmp)
            pd_at_stim = pDetected[stim_idx[1]]
            # Determine if detected
            if rand() < pd_at_stim || (NumAFC > 1 && rand() < afc_chance)
                # Deterimine if the 'correct' interval was selected. Equal to p(detected) + 1/NumAFC
                detection_history[t,p] = true
            end

            # Update rule depends on number of trials
            if sum(detection_history[1:t, p] .== 1) < 3 || sum(detection_history[1:t, p] .== 0) < 3
                if detection_history[t,p] == 1
                    StimAmp -= StepSize
                else
                    StimAmp += StepSize
                end
    
            else # Fit a logistic function and use estimated point as next trial
                r = optimize(x -> ChanceMaximumLikelihood(amplitude_history[1:t, p],
                        detection_history[1:t, p], afc_chance, x),
                        LowerBounds, UpperBounds, PermGuess)
                PermGuess = r.minimizer
                estimated_threshold[t,p] = abs(r.minimizer[2]) / r.minimizer[1]
                StimAmp = estimated_threshold[t,p] + (r.minimizer[1] * randn() * NoiseFactor)
            end
            
            # Validate StimAmp
            StimAmp = round(StimAmp / 2) * 2
            if StimAmp < min_stim_level
                StimAmp = min_stim_level
            elseif StimAmp > max_stim_level
                StimAmp = max_stim_level
            end
        end
    end

    return detection_history, amplitude_history, estimated_threshold
end

"""
ChanceMaximumLikelihood(X::Vector{Int}, Y::Vector{Bool}, Chance::Float64, Coeffs::Vector{Float64})

Custom loss function for logistic regression. Scales probabilities by chance to prevent over-fitting
    to false alarms. Returns the inverted likelihood (loss) such that minimization optimizers can be used.

returns loss
"""
function ChanceMaximumLikelihood(X::Vector{Int64},
                                 Y::Union{BitVector, Vector{Bool}},
                                 Chance::Float64,
                                 Coeffs::Vector{Float64})
    log_odds_projected = X .* Coeffs[1] .+ Coeffs[2] # Project in form y = mx + c
    # Convert to probability while adjusting for chance
    probabilities = (exp.(log_odds_projected) ./ (1 .+ exp.(log_odds_projected))) .* Chance .+ Chance
    # Compute likelihoods
    likelihood_detected = [log(x) for (x, y) in zip(probabilities, Y) if y];
    likelihood_not_detected = [log(1 - x) for (x, y) in zip(probabilities, Y) if !y];
    total_likelihood = sum(likelihood_detected) + sum(likelihood_not_detected)
    return total_likelihood * -1
end