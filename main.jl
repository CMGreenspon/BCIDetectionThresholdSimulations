start_time = time()
using DataFrames, UnicodePlots, StatsBase, Distributions, Statistics, LsqFit, Distributed, NaNStatistics, Base.Threads
include("ConstantStimulationSimulation.jl")

println("Imports complete.")
## Helper functions
jnd2sigma(j::Real) = (1 /quantile(Normal(), 0.75)) * j # Convert JND to σ
sigma2k(σ::Real) = 1.7 / σ; # Convert σ to k
sigmoid(x::Vector, coeffs::Vector) = 1 ./ (1 .+ exp.(-coeffs[1].*(x.-coeffs[2])))
# Hardcoded parameters
valid_stims = collect(2:2:100) # These are the amplitudes that can be given 

##( Define psychophysical curve as cumulative normal distribution
detection_threshold = 50 # microamps
jnd = 10 # microamps
sigma = jnd2sigma(jnd)

psychometric_pdf = Normal(detection_threshold, sigma)
pDetected = cdf(psychometric_pdf, valid_stims)

pDetected_plot = lineplot(valid_stims, pDetected, title="Psychometric Curve", color=:green,
    name = "Cumulative Probability", xlabel = "Amplitude (μA)", ylabel = "p(Detected)",
    width = 80, height = 20, blend = false)
lineplot!(pDetected_plot, [detection_threshold, detection_threshold], 
    [0, 1], color=(169, 169, 169), name = "DT50 = $(detection_threshold)")
lineplot!(pDetected_plot, [detection_threshold, detection_threshold].- jnd/2, [0, 1],
    color = :red, name = "JND = $(jnd)")
    lineplot!(pDetected_plot, [detection_threshold, detection_threshold].+ jnd/2, [0, 1],
    color = :red)
display(pDetected_plot)##)

##( Method of constants 2AFC simulation
num_AFC = 2
num_levels = 8
test_stims, test_stims_pDetected = GetConstantTargets(valid_stims, pDetected, Mode = "Unconstrained")

# Prepare plot
ntrials_constants_plot = lineplot(valid_stims, pDetected, title="# Trials", color=(169, 169, 169),
    name = "Ground Truth", xlabel = "Amplitude (μA)", ylabel = "p(Detected)",
    width = 80, height = 20, blend = false)
# Show for each number of trial
for num_trials in [5, 10, 50]
    pd = ConstantSimulation(test_stims, test_stims_pDetected, num_trials, NumPerms = 1, NumAFC = num_AFC)
    scatterplot!(ntrials_constants_plot, test_stims, vec(pd), name="#T = $(num_trials)", marker=:circle)
end
display(ntrials_constants_plot)##) 

##( Show effect of number of trials on threshold estimate error
max_trials = Int(1e2)
num_perms = Int(1e3)
t_est = zeros(max_trials, num_perms); fill!(t_est, NaN)
Threads.@threads for t = 1:max_trials
    t_est[t,:] = ConstantSimulation(test_stims, test_stims_pDetected, t, NumPerms = num_perms, NumAFC = num_AFC)
end

t_mean = nanmean(t_est, dims = 2)
t_std = nanstd(t_est, dims = 2)
t_error = nanmean((t_est .- detection_threshold).^2, dims = 2);

ntrials_constants_plot2 = lineplot(1:max_trials, vec(t_mean), title="Prediction Variance",
 color=(169, 169, 169), name = "Mean", xlabel = "#Trials", ylabel = "Pred Amplitude",
    width = 80, height = 20, blend = false, ylim=(40, 60), xlim=(5,max_trials))
lineplot!(ntrials_constants_plot2, 1:max_trials, vec(t_mean) .+ vec(t_std),
     color=:blue, name = "STD")
lineplot!(ntrials_constants_plot2, 1:max_trials, vec(t_mean) .- vec(t_std),
     color=:blue)
display(ntrials_constants_plot2)

ntrials_constants_plot3 = lineplot(1:max_trials, vec(t_error), title="Prediction Error",
    color=:blue, xlabel = "#Trials", ylabel = "MSE", xscale=:log10,
    width = 80, height = 20, ylim = (0,100), xlim = (5, max_trials))
for t = [5, 10, 50]
    lineplot!(ntrials_constants_plot3, [t, t], [0, 100], color=(169, 169, 169))
    annotate!(ntrials_constants_plot3, t, 100, string(t), color = (169, 169, 169))
end
display(ntrials_constants_plot3)##)


run_time = time() - start_time