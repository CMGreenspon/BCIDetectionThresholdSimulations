## Import and setup
    using LsqFit, Distributions, GLM, DataFrames, UnicodePlots, StatsBase, NaNStatistics, Base.Threads
    jnd2sigma(j::Real) = (1 /quantile(Normal(), 0.75)) * j # Convert JND to σ
    sigma2k(sigma::Real) = 1.7 / sigma; # Convert σ to k
    sigmoid(x::Vector, coeffs::Vector) = 1 ./ (1 .+ exp.(-coeffs[1].*(x.-coeffs[2])))
    # Hardcoded parameters
    valid_stims = collect(2:2:100) # These are the amplitudes that can be given 
    ## Define psychophysical curve as cumulative normal distribution
    detection_threshold = 50 # microamps
    jnd = 5 # microamps
    sigma = jnd2sigma(jnd)

    psychometric_pdf = Normal(detection_threshold, sigma)
    pDetected = cdf(psychometric_pdf, valid_stims)
    lineplot(valid_stims, pDetected)


##
    include("StaircaseSimulations.jl")
## PEST Testing
    num_afc = 2; afc_chance = 1/num_afc
    StimAmp = rand(valid_stims,1)[1] # Random starting amplitude
    max_trials = 100
    step_size = 10
    target_performance = .75
    wald_value = 1
    min_trials = 4
    MinStepSize = 2
    MaxStepSize = 20
    max_stim_level = maximum(valid_stims)
    min_stim_level = minimum(valid_stims)

    # PEST Rules
    current_dir = -1
    new_dir = -1
    last_rev_was_double = false
    last_step_was_double = false
    sequential_dir_count = 0
    temp_detection_tracker = Int[]

    amplitude_history = fill(NaN, max_trials)
    for t = 1:max_trials
        stim_idx = findall(ValidStims .== StimAmp)
        if length(stim_idx) == 0 # If invalid stim then find nearest valid
            (_, stim_idx) = findmin(abs.(ValidStims .- StimAmp))
            StimAmp = ValidStims[stim_idx[1]]
        end
        amplitude_history[t] = StimAmp
        pd_at_stim = pDetected[stim_idx[1]] # Will error if the StimAmp is not a member of ValidStims
        
        # Determine if detected
        if rand() < pd_at_stim || (num_afc > 1 && rand() < afc_chance)
            # Deterimine if the 'correct' interval was selected. Equal to p(detected) + 1/NumAFC
            push!(temp_detection_tracker, 1)
        else
            push!(temp_detection_tracker, 0)
        end

        # Evaluate Wald-adjusted likelihood, udate stim if criteria is met, reset detection tracker
        if length(temp_detection_tracker) >= min_trials
            observed_detections = sum(temp_detection_tracker)
            expected_detections = length(temp_detection_tracker) * target_performance
            wald_bounds = (expected_detections + wald_value, expected_detections - wald_value)
            if observed_detections > wald_bounds[1] # If more than expected then decrease amp
                new_dir = -1
                StimAmp -= step_size
                temp_detection_tracker = Int[]
            elseif observed_detections < wald_bounds[2]# If less than expected then increase amp
                new_dir = 1
                StimAmp += step_size
                temp_detection_tracker = Int[]
            end
        end

        # Ensure StimAmp is in range and is valid (also count as reversion/early stopping criteria)
        if StimAmp > max_stim_level
            StimAmp = max_stim_level
            # reversion_history[t,p] = new_direction*-1
        elseif StimAmp < min_stim_level
            StimAmp = min_stim_level
            # reversion_history[t,p] = new_direction*-1
        end

        # Determine new update rule - PEST Rules
        if length(temp_detection_tracker) == 0
            if new_dir != current_dir
                current_dir = new_dir
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
    end

    