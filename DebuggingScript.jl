# Debugging script
using DataFrames, UnicodePlots, StatsBase, Distributions, Statistics, LsqFit, Distributed,
      Base.Threads, Revise, Infiltrator, NaNStatistics, GLM
      
## Helper functions
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

# Run the simulation and show an example staircase
    target_p = GetTransformedStaircaseTarget(2, [3,1])
    target_amplitude = quantile(Normal(detection_threshold, sigma), target_p)
    amplitude_history, detection_history, reversion_history, estimated_thresholds, stop_point =
        TransformedStaircaseSimulation(valid_stims,  pDetected, NumPerms=100, MaxReversions = 100)
    
    # Implement GLM fit from staircase
    # first lets remove all nan-values
    nan_idx = findall(isnan.(vec(amplitude_history[:,1])) .== 0)
    amp_temp = amplitude_history[nan_idx,1]

    transformed_staircase_plot = lineplot([1,50], [target_amplitude, target_amplitude], color=(169, 169, 169)
    , ylim = (0, 100), width = 80, height = 20, xlabel="Trial #", ylabel="Stimulus Amplitude (μA)")
    lineplot!(transformed_staircase_plot, 1:size(amp_temp,1), amp_temp)
    scatterplot!(transformed_staircase_plot, 1:size(amp_temp,1), amp_temp)
    display(transformed_staircase_plot)

    RevThresholds = PosthocEstimateStaircaseThreshold(amplitude_history, reversion_history,
     MaxReversions = 20, SkipFirstNReversions = 2, UseMLE = false)
    MLEThresholds, MLESigmas = PosthocEstimateStaircaseThreshold(amplitude_history, reversion_history,
     MaxReversions = 100, UseMLE = true)

    # mean(MLEThresholds)
    # mean(RevThresholds)