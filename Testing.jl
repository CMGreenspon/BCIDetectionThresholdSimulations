## Import 
    using LsqFit, Distributions, GLM, DataFrames, UnicodePlots
    using StatsBase, NaNStatistics, Base.Threads, DataFrames, Optim
    using Printf
    include.(("ConstantSimulations.jl", "StaircaseSimulations.jl"))

## Setup
    jnd2sigma(j::Real) = (1 /quantile(Normal(), 0.75)) * j # Convert JND to σ
    sigma2k(sigma::Real) = 1.7 / sigma; # Convert σ to k
    sigmoid(x::Vector, coeffs::Vector) = 1 ./ (1 .+ exp.(-coeffs[1].*(x.-coeffs[2])))
    # Hardcoded parameters
    valid_stims = collect(2:2:100); # These are the amplitudes that can be given 
    ## Define psychophysical curve as cumulative normal distribution
    detection_threshold = 50; # microamps
    jnd = 12; # microamps
    sigma = jnd2sigma(jnd);

    psychometric_pdf = Normal(detection_threshold, sigma);
    pDetected = cdf(psychometric_pdf, valid_stims);
    l  = lineplot(valid_stims, pDetected);
    lineplot!(l, [detection_threshold, detection_threshold, 0], [0, .5, .5])

## Quest testing
    detection_history, amplitude_history, estimated_threshold = QUESTStaircase(valid_stims,
        pDetected, NumPermutations = 1, MaxTrials = 50)
    lineplot(1:length(amplitude_history), amplitude_history)

##  Asessing different metrics
    # Generate N trials and work get detections for each trial
    afc_chance = 0.5
    num_trials = 100;
    StimIdx = rand(1:length(valid_stims), num_trials);
    StimAmps = valid_stims[StimIdx];
    Detections = (rand(num_trials) .< pDetected[StimIdx]) .| (rand(num_trials) .< afc_chance);

    # Ground truth
    p = lineplot(valid_stims, pDetected);
    scatterplot!(p, StimAmps, Detections);

    # Logistic regression
    df = DataFrame([StimAmps, Detections], ["StimAmp", "Detected"]);
    fm1 = @formula(Detected ~ StimAmp);
    logit1 = glm(fm1, df, Binomial(), ProbitLink());
    fit_coeffs = coef(logit1);
    l1_pred = abs(fit_coeffs[1])/fit_coeffs[2];
    # lineplot!(p, valid_stims, predict(logit1, DataFrame([valid_stims], ["StimAmp"])));

    # Modified logistic regression
    upper = [1.0, 100.0];
    lower = [0.0, -100.0];
    initial_x = [0.1, 0];
    r = optimize(x -> ChanceMaximumLikelihood(StimAmps, Detections, 0.5, x),
        lower, upper, initial_x);
    cml_pred = abs(r.minimizer[2]) / r.minimizer[1]
    
    # LSQ
    sig_fit = curve_fit(sigmoid, StimAmps, Detections, [0.1, 50]);

    # Modified LSQ
    function ChanceLSQ(X, Y, Chance, Coeffs)
        YPred = sigmoid(X, Coeffs)
        resids = Y .- YPred
        gain = 1 .- (Y .* Chance)
        return sum(abs.(resids .* gain))
    end

    r = optimize(x -> ChanceLSQ(StimAmps, Detections, 0.5, x),
        lower, upper, initial_x);
    sig_y = sigmoid(valid_stims, r.minimizer);
    lineplot!(p, valid_stims, sig_y);
    clsq_pred = abs(r.minimizer[2]) / r.minimizer[1]

    # Outputs
    display(p)
    @printf("Logistic regression prediction = %0.2f\n", l1_pred);
    @printf("ChanceMaximumLikelihood prediction = %0.2f\n", cml_pred);
    @printf("Least-square regression prediction = %0.2f\n", sig_fit.param[2]);
    @printf("Custom least-square regression prediction = %0.2f\n", clsq_pred);

## Looking at empirical cumulative plots
    detection_history, amplitude_history, estimated_threshold = QUESTStaircase(valid_stims,
        pDetected, NumPermutations = 1, MaxTrials = 100, NoiseFactor = 20)
    lineplot(1:length(amplitude_history), amplitude_history, ylim = (0, 100))

    # Chunk amplitudes
    Xe = LinRange(percentile(amplitude_history[:,1], 5)-1, percentile(amplitude_history[:,1], 95)+1, 7);
    Xm = Xe[1:end-1] .+ (Xe[2] - Xe[1]) / 2;
    weight, bin_idx = histcountindices(amplitude_history, Xe);
    pd = [mean(detection_history[bin_idx .== x, 1]) for x in collect(range(1, maximum(bin_idx)))];
    pd = (pd .- 0.5) ./ 0.5
    sig_fit = curve_fit(sigmoid, [0;Xm;100], vec([0;pd;1]), [0.1, mean(Xm)])
    p = lineplot(valid_stims, pDetected);
    scatterplot!(p, Xm, pd, color = :cyan);
    display(p)