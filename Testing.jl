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
    num_trials = Int(1e2);
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
    lineplot!(p, valid_stims, predict(logit1, DataFrame([valid_stims], ["StimAmp"])));

    # Modified logistic regression
    upper = [1.0, 100.0];
    lower = [0.0, -100.0];
    initial_x = [0.1, 0];
    r = optimize(x -> ChanceMaximumLikelihood(amplitude_history[:,1], detection_history[:,1], 0.5, x),
        lower, upper, initial_x);
    cml_pred = abs(r.minimizer[2]) / r.minimizer[1]
    
    # LSQ
    sig_fit = curve_fit(sigmoid, StimAmps, Detections, [0.1, 50]);
    sig_y = sigmoid(valid_stims, sig_fit.param);
    lineplot!(p, valid_stims, sig_y);

    # Outputs
    display(p)
    @printf("Logistic regression prediction = %0.2f", l1_pred);
    @printf("ChanceMaximumLikelihood prediction = %0.2f", cml_pred);
    @printf("Least-square regression prediction = %0.2f", sig_fit.param[2]);

## Custom least squares
function ChanceLSQ(X, Y, Chance, Coeffs)

end