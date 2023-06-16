## Import 
    using LsqFit, Distributions, GLM, DataFrames, UnicodePlots, StatsBase, NaNStatistics, Base.Threads, DataFrames
    include.(("ConstantSimulations.jl", "StaircaseSimulations.jl"))

## Setup
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
    l  = lineplot(valid_stims, pDetected);
    lineplot!(l, [detection_threshold, detection_threshold, 0], [0, .5, .5])
   

## Quest testing
max_trials = 100;
NumAFC = 2;
afc_chance = 1 / NumAFC;
TrackingDF = DataFrame([Int8[], Int8[]], ["StimAmp", "Detected"])
chance_sigmoid(x::Vector, coeffs::Vector) = (1 ./ (1 .+ exp.(-coeffs[1].*(x.-coeffs[2])))) .* coeffs[3] .+ coeffs[3]
chance_linear(x::Vector, coeffs::Vector) = (coeffs[1] .* (x .+ coeffs[2])) .+ coeffs[3]
# StimAmp = detection_threshold
StimAmp = rand(valid_stims,1)[1];
StepSize = 10;
t = 1;
while t < max_trials + 1
    stim_idx = findall(valid_stims .== StimAmp)
    pd_at_stim = pDetected[stim_idx[1]]
    # Determine if detected
    if rand() < pd_at_stim || (NumAFC > 1 && rand() < afc_chance)
        # Deterimine if the 'correct' interval was selected. Equal to p(detected) + 1/NumAFC
        DTLogit = 1
    else
        DTLogit =  0
    end
    push!(TrackingDF, [convert(Int8, StimAmp), convert(Int8, DTLogit)])

    # Update rule depends on number of trials
    if sum(TrackingDF.Detected .== 1) < 3 || sum(TrackingDF.Detected .== 0) < 3
        if DTLogit == 1
            StimAmp -= StepSize
        else
            StimAmp += StepSize
        end

    else # Fit a logistic function and use estimated point as next trial

        # sig_fit = curve_fit(chance_sigmoid, TrackingDF.StimAmp, TrackingDF.Detected,
        #  [0.1, mean(TrackingDF.StimAmp), afc_chance], lower = [0, 0, afc_chance], upper = [1, 100, afc_chance])
        # StimAmp = sig_fit.param[2] + randn() * exp(sig_fit.param[1]) * 2

        # logit = glm(fm1, TrackingDF, Binomial(), LogitLink())
        # StimAmp = ((0.75 - coef(logit)[1]) / coef(logit)[2]) + randn() * coef(logit)[1] / 10
        
        lin_fit = curve_fit(chance_linear, log.(TrackingDF.StimAmp), TrackingDF.Detected,
         [0.1, mean(log.(TrackingDF.StimAmp)), afc_chance], lower = [0, -100, afc_chance], upper = [1, 100, afc_chance])
        
    end

    StimAmp = round(StimAmp / 2) * 2
    if StimAmp < valid_stims[1]
        StimAmp = valid_stims[1]
    elseif StimAmp > valid_stims[end]
        StimAmp = valid_stims[end]
    end
    t += 1
end

lineplot(range(1,max_trials), TrackingDF.StimAmp, ylim=(0,100))
logit = glm(fm1, TrackingDF, Binomial(), LogitLink())
println("Predicted Threshold = $((0.775 - coef(logit)[1]) / coef(logit)[2])")
##
sig_fit = curve_fit(chance_sigmoid, TrackingDF.StimAmp, TrackingDF.Detected,
         [0.1, mean(TrackingDF.StimAmp), afc_chance], lower = [0, 0, afc_chance], upper = [5, 100, afc_chance])
println("Predicted Threshold = $(sig_fit.param[2])")
fm1 = @formula(Detected ~ StimAmp)
logit1 = glm(fm1, TrackingDF, Binomial(), ProbitLink())


## Bayesian testing
# Generate N trials and work get detections for each trial
afc_chance = 0.5
num_trials = Int(1e6);
StimIdx = rand(1:length(valid_stims), num_trials);
StimAmps = valid_stims[StimIdx];
Detections = (rand(num_trials) .< pDetected[StimIdx]) .| (rand(num_trials) .< afc_chance);

# USA = sort(unique(StimAmps))
# pbin = [mean(Detections[StimAmps .>= x]) for x in USA]
temp_df = DataFrame(StimAmp = StimAmps, Detections = Detections)
fm = @formula(Detections ~ StimAmp)
logit = glm(fm, temp_df, Binomial(), LogitLink())
## Testing logit predict
l = lineplot(valid_stims, convert.(Float64, predict(logit1, DataFrame(StimAmp = valid_stims))), height = 30, width = 80);
lineplot!(l, [detection_threshold, detection_threshold, 0], [0, .75, .75]);
lineplot!(l, [detection_threshold, detection_threshold, 0], [0, .5, .5]);
lineplot!(l, [37, 37, 0], [0, .5, .5])

