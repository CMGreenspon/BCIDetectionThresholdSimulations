start_time = time()
using DataFrames, UnicodePlots, StatsBase, Distributions, Statistics, LsqFit, Distributed,
      NaNStatistics, Base.Threads, Revise
include.(("ConstantSimulations.jl", "StaircaseSimulations.jl"))

println("Imports complete.")
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

    pDetected_plot = lineplot(valid_stims, pDetected, title="Psychometric Curve", color=:green,
        name = "Cumulative Probability", xlabel = "Amplitude (μA)", ylabel = "p(Detected)",
        width = 80, height = 20, blend = false)
    lineplot!(pDetected_plot, [detection_threshold, detection_threshold], 
        [0, 1], color=(169, 169, 169), name = "DT50 = $(detection_threshold)")
    lineplot!(pDetected_plot, [detection_threshold, detection_threshold].- jnd, [0, 1],
        color = :red, name = "JND = $(jnd)")
        lineplot!(pDetected_plot, [detection_threshold, detection_threshold].+ jnd, [0, 1],
        color = :red)
    display(pDetected_plot)
    

## Method of constants
    test_stims, test_stims_pDetected = GetConstantTargets(valid_stims, pDetected, Mode = "Unconstrained")

    # Prepare plot
    ntrials_constants_plot = lineplot(valid_stims, pDetected, title="# Trials", color=(169, 169, 169),
        name = "Ground Truth", xlabel = "Amplitude (μA)", ylabel = "p(Detected)",
        width = 80, height = 20, blend = false)
    # Show for each number of trial
    for num_trials in [5, 10, 50]
        _, pd = ConstantSimulation(test_stims, test_stims_pDetected, num_trials, NumPerms = 1)
        scatterplot!(ntrials_constants_plot, test_stims, vec(pd), name="#T = $(num_trials)", marker=:circle)
    end
    display(ntrials_constants_plot)

## Show effect of number of trials on threshold estimate error
    max_trials = Int(1e2)
    num_perms = Int(1e3)
    t_est = zeros(max_trials, num_perms); fill!(t_est, NaN)
    Threads.@threads for t = 1:max_trials
        t_est[t,:], _ = ConstantSimulation(test_stims, test_stims_pDetected, t, NumPerms = num_perms)
    end

    t_mean_uconsts = nanmean(t_est, dims = 2)
    t_std_uconsts = nanstd(t_est, dims = 2)
    t_error_uconsts = nanmean((t_est .- detection_threshold).^2, dims = 2);

    ntrials_constants_plot2 = lineplot(1:max_trials, vec(t_mean_uconsts), title="Prediction Variance",
    color=(169, 169, 169), name = "Mean", xlabel = "#Trials/Intensity", ylabel = "Pred Amplitude",
        width = 80, height = 20, blend = false, ylim=(40, 60), xlim=(5,max_trials))
    lineplot!(ntrials_constants_plot2, 1:max_trials, vec(t_mean_uconsts) .+ vec(t_std_uconsts),
        color=:blue, name = "STD")
    lineplot!(ntrials_constants_plot2, 1:max_trials, vec(t_mean_uconsts) .- vec(t_std_uconsts),
        color=:blue)
    display(ntrials_constants_plot2)

    ntrials_constants_plot3 = lineplot(1:max_trials, vec(t_error_uconsts), title="Prediction Error",
        color=:blue, xlabel = "#Trials/Intensity", ylabel = "MSE", xscale=:log10,
        width = 80, height = 20, ylim = (0,100), xlim = (5, max_trials))
    for t = [5, 10, 50]
        lineplot!(ntrials_constants_plot3, [t, t], [0, 100], color=(169, 169, 169))
        annotate!(ntrials_constants_plot3, t, 100, string(t), color = (169, 169, 169))
    end
    display(ntrials_constants_plot3)

## Transformed Staircase
    t = time()
    # Run the simulation and show an example staircase
    amplitude_history, detection_history, reversion_history, estimated_thresholds, stop_point =
        TransformedStaircaseSimulation( valid_stims,  pDetected)
    transformed_staircase_plot = lineplot(1:size(amplitude_history,1), vec(amplitude_history[:,1]), ylim = (0, 100))
    display(transformed_staircase_plot)
    
    # Evaluate what the staircase would have predicted for different criteria
    max_reversions = 100
    t_est = zeros(max_reversions, num_perms); fill!(t_est, NaN)
    t_stop = zeros(max_reversions, num_perms); fill!(t_est, NaN)
    Threads.@threads for mr = 1:max_reversions
        _, _, _, t_est[mr,:], t_stop[mr,:]=
            TransformedStaircaseSimulation(valid_stims,  pDetected, MaxReversions = mr, NumPerms = num_perms)
    end

    t_mean_staircase = nanmean(t_est, dims = 2)
    t_std_staircase = nanstd(t_est, dims = 2)
    t_error_staircase = nanmean((t_est .- detection_threshold).^2, dims = 2)

    nreversions_staircase_plot = lineplot(1:max_reversions, vec(t_mean_staircase), title="Prediction Variance",
    color=(169, 169, 169), name = "Mean", xlabel = "#Reversions", ylabel = "Pred Amplitude",
        width = 80, height = 20, blend = false, ylim=(40, 60), xlim=(1,max_reversions))
    lineplot!(nreversions_staircase_plot, 1:max_reversions, vec(t_mean_staircase) .+ vec(t_std_staircase),
        color=:blue, name = "STD")
    lineplot!(nreversions_staircase_plot, 1:max_reversions, vec(t_mean_staircase) .- vec(t_std_staircase),
        color=:blue)
    display(nreversions_staircase_plot)

    nreversions_staircase_plot2 = lineplot(1:max_reversions, vec(t_error_staircase), title="Prediction Error",
        color=:blue, xlabel = "#Reversions", ylabel = "MSE", xscale=:log10,
        width = 80, height = 20, ylim = (0,100), xlim = (5, max_reversions))
    for t = [5, 10, 50]
        lineplot!(nreversions_staircase_plot2, [t, t], [0, 100], color=(169, 169, 169))
        annotate!(nreversions_staircase_plot2, t, 100, string(t), color = (169, 169, 169))
    end
    display(nreversions_staircase_plot2)
    dt = time() - t

## Comparing constants & staircase
    # Because number of reversions != # trials we need to convert the indices
    # Convert to vec for Histogram function
    t_stop_vec = vec(t_stop) 
    t_est_vec = vec(t_est)
    h = fit(Histogram, t_stop_vec, 1:max_trials) 
    # This gives up the bin index (== t_stop) that we can use on t_est
    indmap = StatsBase.binindex.(Ref(h), t_stop_vec)
    #t_est_sorted = fill(Float64[], max_trials, 1)
    t_mean_staircase_sorted = zeros(max_trials)
    t_std_staircase_sorted = zeros(max_trials)
    t_error_staircase_sorted = zeros(max_trials)
    for t = 1:max_trials
        t_idx = findall(indmap .== t)
        #t_est_sorted[t] = t_est[t_idx]
        t_mean_staircase_sorted[t] = mean(t_est[t_idx])
        t_std_staircase_sorted[t] = std(t_est[t_idx])
        t_error_staircase_sorted[t] = mean((t_est[t_idx].- detection_threshold).^2)
    end

    y_ul = 1000
    error_comparison_plot = lineplot([5,5], [1,y_ul], color=(169, 169, 169), title="Prediction Error",
        xlabel = "#Trials", ylabel = "MSE", xscale=:log10, yscale=:log10, width = 80, height = 20, ylim = (10,y_ul),
        xlim = (5, max_trials*2))
    annotate!(error_comparison_plot, 5, y_ul, string(5), color = (169, 169, 169))
    for t = [10, 50]
        lineplot!(error_comparison_plot, [t, t], [1, y_ul], color=(169, 169, 169))
        annotate!(error_comparison_plot, t, y_ul, string(t), color = (169, 169, 169))
    end
    lineplot!(error_comparison_plot, 1:max_trials, t_error_staircase_sorted, color=:green, name = "Staircase")
    lineplot!(error_comparison_plot, collect(1:max_trials).*length(test_stims), vec(t_error_uconsts),
        color=:blue, name = "Unconstrained Constants")
    display(error_comparison_plot)
 
## End
run_time = round(time() - start_time, digits=2)
println("Total run time = $(run_time) s")