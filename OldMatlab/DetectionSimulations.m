%%% Detection Threshold Simulations & Analysis
% 1a. Simulate detection thresolds for single interval and forced choice tasks up to 5AFA
% 1b. Determine how accurate predictions are for each point of the sigmoid as well as the
% JND for a given number of observations per condition
% 1c. Check how Alternate Force Choice (AFC) alters setup. Use signal detection theory for
% this
% 1d. Relationship between number of choices and accuracy metrics
% 2a. Simulate a simple staircase method (without constraints) to determine time to
% conversion, absolute error at conversion point, and variance at conversion
% 2b. Compare stopping criteria: number of trials, number of reversions, average over
% previous trials, average of previous conversions
% 2c. Add minimum step size constraint to determine how it effects previous
% parameters - in particular relationship between step size and threshold
% 2d. Add 2-5 AFC options to examine if this alters any metric
% 2e. Relationship between step ratio and minimum step size to time to convergence
% 3a. Simulate transformed pyramids (5-1 : 1-5)
% 3b. Determine error and convergence metrics
% 4. Simulate 2 step process
% 4a. Determine optimal convergence method based on previous analyses
% 4b. Perform initial convergence step and then use results from 1b to perform a full
% psychometric sweep
% 5. Adaptive sweep method to determine bounds and center point all in one go
% Bonus
% B1. How does the range of intitial values effect convergence time (i.e. random start vs
% within 10 uA of start)
% B2. What happens if you increase step size on incorrect trials/upward reversions

%% Helper functions & Setup
% Sigmoid function for estimation
sigfun = @(c,x) 1./ (1 + exp(-c(1).*(x-c(2)))); % c(1) = rate of change, c(2) = x-offset
invsig = @(c,y) (log(1/y - 1)/-c(1)) + c(2);
% Psychometric helper functions
jnd2sig = @(j) (1/norminv(.75))*j;
getpdetect = @(stim, threshold, sigma) normcdf(stim, threshold, sigma);

opts = optimset('Display','off'); % Disable reporting for lsqcurvefit
SetFont('Arial', 16)

%% 1a Simple detection task
n_perms = 1e2; % Number of times to repeat each simulation
max_repeats = 1e2; % Number of stimuli at each intensity

intensity = 1:0.01:100;
threshold = 50;
jnd = threshold * .4;
sigma = jnd2sig(threshold * .4);

ground_truth = getpdetect(intensity, threshold, sigma); % Create CDF/sigmoid
stims = threshold-(5*4) : 5 : threshold+(5*4); % 9 stimuli centered on threshold in increments of 5

% Get probability of detection at each stimulus intensity
intensity_idx = zeros([1,length(stims)]);
for s = 1:length(stims)
    [~, intensity_idx(s)] = min(abs(intensity-stims(s)));
end
p_detect = ground_truth(intensity_idx)';

[repeat_mse, sig_mse, sig_t, sig_jnd] = deal(zeros([n_perms, max_repeats]));
[wb, pfwb_update]  = ParforWaitbar('1a. Simple detection simulations', max_repeats*n_perms); %send(pfwb_update, 0);

parfor i = 1:max_repeats
    for f = 1:n_perms
        % Get draws - uniform distribution in this case
        draw = rand([length(stims), i]);
        detections = p_detect > draw;
        detections = mean(detections,2);
        % Measure MSE at each point
        repeat_mse(f,i) = mean(sum((detections - p_detect).^2));
        % Fit a sigmoid and then measure MSE from sigmoid
        [sig_c, ~] = lsqcurvefit(sigfun,[rand(1),randi([10,90])], stims, detections', [0, 10],[1, 90], opts);
        sig_y = sigfun(sig_c, stims);
        sig_mse(f,i) = mean(sum((sig_y - p_detect').^2));
        sig_t(f,i) = sig_c(2);
        % Determine estimated JND
        sig_jnd(f,i) = (invsig(sig_c,.75) - invsig(sig_c,.25))/2;
        send(pfwb_update, 0);
    end
end

%% 1a Plotting
colors = lines(5);
marker_size = 40;

% For plotting only
draw = rand([length(stims), max_repeats]);
detections = p_detect > draw;
detections_5 = mean(detections(:,randi(max_repeats,[5,1])),2);
detections_10 = mean(detections(:,randi(max_repeats,[10,1])),2);
detections_50 = mean(detections(:,randi(max_repeats,[50,1])),2);
detections_100 = mean(detections(:,randi(max_repeats,[100,1])),2);

fig1 = figure('Name', 'Simple Detection');
f1_sp1 = subplot(2,3,[1,4], 'Parent', fig1); hold on
    % True distribution
    plot(intensity, ground_truth, 'Color', [0.6 0.6 0.6], 'LineStyle', '--', 'Parent', f1_sp1)
    plot([threshold,threshold,0], [0, 0.5, 0.5], 'Color', [0.6 0.6 0.6], 'LineStyle', ':', 'Parent', f1_sp1)
    % Observed distribution
    scatter(stims, detections_5, marker_size, 'MarkerEdgeColor', colors(1,:),...
        'MarkerFaceColor', colors(1,:), 'MarkerFaceAlpha', .3, 'Parent', f1_sp1) 
    scatter(stims, detections_10, marker_size, 'MarkerEdgeColor', colors(2,:),...
        'MarkerFaceColor', colors(2,:), 'MarkerFaceAlpha', .3, 'Parent', f1_sp1) 
    scatter(stims, detections_100, marker_size, 'MarkerEdgeColor', colors(3,:),...
        'MarkerFaceColor', colors(3,:), 'MarkerFaceAlpha', .3, 'Parent', f1_sp1) 

    leg_text = ColorText({'5', '10', '100', 'Repetitions'}, [colors(1:3,:); [.6 .6 .6]]);
    text(5, 0.975, leg_text, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left')

    ylabel('p(Detection)')
    xlabel('Amplitude')

x = 1:max_repeats;
f1_sp2 = subplot(2,3,[2,5], 'Parent', fig1); hold on
    AlphaLine(x, repeat_mse', colors(4,:), 'ErrorType', 'Percentile',...
        'Percentiles', [25, 75], 'Parent', f1_sp2)
    AlphaLine(x, sig_mse', colors(5,:), 'ErrorType', 'Percentile',...
        'Percentiles', [25, 75], 'Parent', f1_sp2)

    set(gca, 'XScale', 'log')
    xlabel('# Trails/Intensity')
    ylabel('MSE')

    leg_text = ColorText({'Raw', 'Sigmoid'}, colors(4:5,:));
    ylim([0 2])
    text(max_repeats*0.95, 0.975*2, leg_text, 'VerticalAlignment', 'top',...
        'HorizontalAlignment', 'right', 'Parent', f1_sp1)

f1_sp3 = subplot(2,3,3, 'Parent', fig1); hold on
    plot([1,max_repeats], [threshold, threshold], 'Color', [0.6 0.6 0.6],...
        'LineStyle', ':', 'Parent', f1_sp3)
    AlphaLine(x, sig_t', colors(5,:), 'ErrorType', 'Percentile',...
        'Percentiles', [25, 75], 'Parent', f1_sp3)
    ylabel('Threshold'); xticklabels({})
    set(gca, 'XScale', 'log')
    ylim([threshold-jnd/2 threshold+jnd/2])

f1_sp4 = subplot(2,3,6, 'Parent', fig1); hold on
    plot([1,max_repeats], [jnd, jnd], 'Color', [0.6 0.6 0.6], 'LineStyle',...
        ':', 'Parent', f1_sp4)
    AlphaLine(x, sig_jnd', colors(5,:), 'ErrorType', 'Percentile',...
        'Percentiles', [25, 75], 'Parent', f1_sp4)
    ylabel('JND'); xlabel('# Trails/Intensity')
    set(gca, 'XScale', 'log')
    ylim([0 jnd*2])

set(fig1, 'Units', 'pixels', 'Position', OSScreenSize([35,15], 'cm', 1));
drawnow

%% AFC: Trial count/performance
[afc_repeat_mse, afc_sig_mse, afc_sig_t, afc_sig_jnd] = deal(zeros([n_perms, max_repeats]));
[wb, pfwb_update]  = ParforWaitbar('AFC simple detection simulations', max_repeats*n_perms); %send(pfwb_update, 0);

parfor i = 1:max_repeats
    for f = 1:n_perms
        % Get draws - uniform distribution in this case
        draw = rand([length(stims), i]);
        detections = p_detect > draw;
        detections = mean(detections,2);
        % Measure MSE at each point
        repeat_mse(f,i) = mean(sum((detections - p_detect).^2));
        % Fit a sigmoid and then measure MSE from sigmoid
        [sig_c, ~] = lsqcurvefit(sigfun,[rand(1),randi([10,90])], stims, detections', [0, 10],[1, 90], opts);
        sig_y = sigfun(sig_c, stims);
        sig_mse(f,i) = mean(sum((sig_y - p_detect').^2));
        sig_t(f,i) = sig_c(2);
        % Determine estimated JND
        sig_jnd(f,i) = (invsig(sig_c,.75) - invsig(sig_c,.25))/2;
        send(pfwb_update, 0);
    end
end

if stim_draw > noise_draws(t) || (num_afc > 1 && rand(1) < chance)

%% 1b Signal detection theory and alternate forced choice implementation
d_prime = -3.5:.01:3.5;
d_prime_intensities = threshold + (sigma/2.*d_prime);
noise_mean = threshold + (sigma/2*-8);

% The 'noise' distribution will always be approximately -4 standard deviations from
% the detection threshold. The mean of the noise distribution is thus
% dependent on the absolute threshold (t) and the variance of the noise
% distribution (which is also the JND at threshold)
% Therefore:
% noise_mean = threshold + (sigma/2*-8);
% Ensure that we are stimulating within the range - dense sampling
sampling = 'stim'; % dense or stim
if strcmp(sampling, 'dense')
    int_idx = d_prime_intensities  > min(intensity) & d_prime_intensities < max(intensity);
    d_prime = d_prime(int_idx);
    d_prime_intensities = d_prime_intensities(int_idx);
elseif strcmp(sampling, 'stim')
    % Get the d' for each stimulus
    d_intensity_idx = zeros([1,length(stims)]);
    for s = 1:length(stims)
        [~, d_intensity_idx(s)] = min(abs(d_prime_intensities-stims(s)));
    end
    d_prime = d_prime(d_intensity_idx);
    d_prime_intensities = d_prime_intensities(d_intensity_idx);
end

% On each trial there will be a draw from the noise(s) and a draw from the
% noise plus signal. The 'subject' will pick the value with the higher
% draw only if it is above therhsold. If more than one draw is above threshold then they
% would randomly pick from the intervals above threshold.
% Timings: 1e2 = 90s, 1e3 = 2300

n_reps = max_repeats*n_perms;
max_afc = 5;

[wb, pfwb_update]  = ParforWaitbar('1b. SDT AFC Simulations', max_repeats*(max_afc-1)); %send(pfwb_update, 0);

[afc_sig_mse_cell, afc_sig_t_cell, afc_sig_jnd_cell, afc_detections] = deal(cell([max_afc-1,1]));
parfor num_afc = 2:max_afc
    detections = zeros([length(d_prime), n_reps]);
    for d = 1:length(d_prime)
        % Draw noise
        noise_draw = normrnd(noise_mean, sigma, [num_afc-1,n_reps]);

        noise_plus_signal_mean = threshold + (sigma/2*d_prime(d));
        % Shouldn't be possible but because of way d' is calculated this can occur
        if noise_plus_signal_mean < noise_mean 
            noise_plus_signal_mean = noise_mean;
        end

        noise_plus_signal_draw = normrnd(noise_plus_signal_mean, sigma, [1,n_reps]);
        
        % This can be performed with indexing but the trial logic is easier to read here
        % and it takes aproximately 1 second to do 1k sims, the rest of the time is spent
        % on lsqcurvefit
        for r = 1:n_reps
            % Decision logic:
            % The subject is only 'guaranteed' to know in which interval the
            % stimulus is in if the Noise+Signal draw is above threshold while the Noise only
            % draw is below threshold. 
            % Therefore:

            % The subject always detects when stim is over and all noises are under
            if noise_plus_signal_draw(r) > threshold && all(noise_draw(:,r) < threshold)
                detections(d,r) = 1;
            % If the stim is over but so are some noises then the subject randomly chooses
            % from those intervals that are above threshold
            elseif noise_plus_signal_draw(r) > threshold && any(noise_draw(:,r) > threshold)
                chance = 1 / (1 + sum(noise_draw(:,r) > threshold));
                if rand(1) < chance
                    detections(d,r) = 1;
                end
            % If stim is under and any noise is above then they will never detect it
            elseif noise_plus_signal_draw(r) < threshold && any(noise_draw(:,r) > threshold)
                detections(d,r) = 0;
            % Finally, if all draws are under detection then there is a chance they will
            % select the correct interval
            elseif noise_plus_signal_draw(r) < threshold && all(noise_draw(:,r) < threshold)
                chance = 1 / num_afc;
                if rand(1) < chance
                    detections(d,r) = 1;
                end
            end % Noise versus Signal
        end % Reps
    end % d'
    
    chance = 1 / num_afc;
    [afc_repeat_mse, afc_sig_mse, afc_sig_t, afc_sig_jnd] = deal(zeros([n_perms, max_repeats]));
    for i = 1:max_repeats
        for f = 1:n_perms
            % Take a random selection of the trials
            temp_idx = randperm(max_repeats*n_perms);
            temp_idx = temp_idx(1:i);
            temp_detections = detections(:,temp_idx);
            % Get average across repeats
            temp_detections = mean(temp_detections,2);
            % Rescale for chance
            temp_detections = (temp_detections-chance)/(1-chance);
            % Measure MSE at each point
            afc_repeat_mse(f,i) = mean(sum((temp_detections - p_detect).^2));
            % Fit a sigmoid and then measure MSE from sigmoid
            [sig_c, ~] = lsqcurvefit(sigfun,[rand(1),randi([10,90])], stims, temp_detections', [0, 10],[1, 90], opts);
            sig_y = sigfun(sig_c, stims);
            afc_sig_mse(f,i) = mean(sum((sig_y - p_detect').^2));
            afc_sig_t(f,i) = sig_c(2);
            % Determine estimated JND
            afc_sig_jnd(f,i) = (invsig(sig_c,.75) - invsig(sig_c,.25))/2;
        end
        send(pfwb_update, 0)
    end
    
    % Assign to cell array for later plotting
    afc_detections{num_afc-1} = detections;
    afc_sig_mse_cell{num_afc-1} = afc_sig_mse;
    afc_sig_t_cell{num_afc-1} = afc_sig_t;
    afc_sig_jnd_cell{num_afc-1} = afc_sig_jnd;
end


%% 1b Plotting
colors = lines(max_afc-1);
marker_size = 40;

fig2 = figure('Name', 'AFC Signal Detection Simulations');
f2_sp1 = subplot(2,3,[1,4], 'Parent', fig2); hold on
    % True distribution
    plot(intensity, ground_truth, 'Color', [0.6 0.6 0.6], 'LineStyle', '--',...
        'Parent', f2_sp1)
    plot([threshold,threshold,0], [0, 0.5, 0.5], 'Color', [0.6 0.6 0.6],...
        'LineStyle', ':', 'Parent', f2_sp1)

    [afc_detections_for_plotting, labels] = deal(cell([max_afc-1,1]));
    num_repeats_for_plotting = 25;
    for i = 2:max_afc
        chance = 1 / i;
        rand_idx = randperm(n_reps);
        rand_idx = rand_idx(1:num_repeats_for_plotting);
        mean_detections = mean(afc_detections{i-1}(:,rand_idx),2);
        afc_detections_for_plotting{i-1} = (mean_detections-chance)/(1-chance);
        afc_detections_for_plotting{i-1}(afc_detections_for_plotting{i-1} < 0) = 0;
        scatter(stims, afc_detections_for_plotting{i-1}, marker_size,...
                'MarkerEdgeColor', colors(i-1,:), 'MarkerFaceColor', colors(i-1,:), 'MarkerFaceAlpha', .3)
        labels{i-1} = num2str(i);
    end
    labels = [{['AFC - ', num2str(num_repeats_for_plotting), ' repeats']}; labels];
    leg_text = ColorText(labels, [[.6 .6 .6]; colors]);
    text(5, 0.975, leg_text, 'VerticalAlignment', 'top',...
        'HorizontalAlignment', 'left', 'Parent', f2_sp1)

    ylabel('p(Detection)')
    xlabel('Amplitude')

x = 1:max_repeats;
f2_sp2 = subplot(2,3,2); hold on
    plot([10 10], [0 1.5], 'Color', [.6 .6 .6], 'LineStyle', '--')
    for i = 2:max_afc
        AlphaLine(x, afc_sig_mse_cell{i-1}', colors(i-1,:), 'ErrorType',...
            'Percentile', 'Percentiles', [25 75], 'Parent', f2_sp2)
    end
    set(f2_sp2, 'XScale', 'log')
    ylabel('MSE', 'Parent', f2_sp2); ylim([0 1.5])
    xlabel('MSE @ 10 Trails/Intensity', 'Parent', f2_sp2)
    xticklabels({})

f2_sp3 = subplot(2,3,5); hold on
    for i = 2:max_afc
        SymphonicBeeSwarm(i, afc_sig_mse_cell{i-1}(:,10), colors(i-1,:),...
            30, 'CenterMethod', 'median', 'Parent', f2_sp3)
    end
    ylabel('MSE @ 10 Trails/Intensity', 'Parent', f2_sp3)
    xlabel('# AFC', 'Parent', f2_sp3); xlim([1.5 max_afc+.5])
    ylim([0 1]); 


f2_sp4 = subplot(2,3,3); hold on
    plot([1,max_repeats], [threshold, threshold], 'Color', [0.6 0.6 0.6],...
        'LineStyle', ':', 'Parent', f2_sp4)
    for i = 2:max_afc
        AlphaLine(x, afc_sig_t_cell{i-1}', colors(i-1,:),'ErrorType',...
            'Percentile', 'Percentiles', [25 75], 'Parent', f2_sp4)
    end
    ylabel('Threshold', 'Parent', f2_sp4); 
    set(f2_sp4, 'XScale', 'log')
    ylim([threshold-jnd/2 threshold+jnd/2])

f2_sp5 = subplot(2,3,6); hold on
    for i = 2:max_afc
        plot([1:max_repeats], std(afc_sig_t_cell{i-1},[],1))
    end
    set(f2_sp5, 'XScale', 'log')
    ylabel('Std. Threshold', 'Parent', f2_sp5);
    xlabel('# Trails/Intensity', 'Parent', f2_sp5); 
        
set(fig2, 'Units', 'pixels', 'Position', OSScreenSize([35,15], 'cm', 1));
drawnow

%% 2a Staircase Simulation
% Target threshold is dependent upon #Intervals, criterion, and consecutive constraints
% For 1 interval with consecutive constraints p = nthroot(.5, [u/d])
% [1,1] = 50%, [2,1] = 70%, [3,1] = 80%, [4,1] = 84%, [5,1] = 87%
%              [1,2] = 30%, [1,3] = 20%, [1,4] = 16%, [1,5] = 13%
% --> Percent correct is equivalent to p(Detect)

% For 2-intervals (chance = 0.5) with consecutive constraints p = (nthroot(.5, [u/d]) - chance) / chance
% [1,1] = 50% p(.0), [2,1] = 70% p(0.4), [3,1] = 80% p(0.59), [4,1] = 84% p(0.68), [5,1] = 87% p(0.74)

example_plot = true;
init_step_size = 20;
max_trials = 500;
trial_vec = 1:max_trials;

% Step size incrementation
step_method = 'Convergent';
min_reversions = 0; % <-- not implemented
% Convergent (1DB power ratio = 1.259; amplitude ratio = 1.122)
pos_step_coeff = .75;
neg_step_coeff = 1;
% Fixed
pos_step = 4;
neg_step = 2;

% Task design
num_afc = 2;
chance = 1 / num_afc;
consecutive_criterion = [3,1];
if num_afc > 1
    if consecutive_criterion(1) > consecutive_criterion(2) && consecutive_criterion(2) == 1
        target_lvl = (nthroot(.5, consecutive_criterion(1)/consecutive_criterion(2)) - chance) / chance;
    elseif consecutive_criterion(2) > consecutive_criterion(1) && consecutive_criterion(1) == 1
        target_lvl = (1-nthroot(.5, consecutive_criterion(2)/consecutive_criterion(1))) * chance;
    end
else
    if consecutive_criterion(1) > consecutive_criterion(2)
       target_lvl = consecutive_criterion(1) / sum(consecutive_criterion);
    else
        target_lvl = consecutive_criterion(2) / sum(consecutive_criterion);
    end
end

% Available stimulus levels
min_step_size = 2;
max_step_size = 20;
valid_stim_values = 2:2:98;
force_valid_stim_values = true;

t2 = valid_stim_values;
t2_colors = lines(length(t2));

[wb, pfwb_update]  = ParforWaitbar('AFC Staircase Simulations', n_perms*length(t2)); %

if example_plot
    fig3 = figure('Name', 'Staircase Examples'); 
    f2_a1 = axes('Parent', fig3); hold on
    example_t2 = 1:length(t2);
    example_t2 = example_t2([1:round(length(t2)/5):end]);
end
[staircase_detections, staircase_amplitudes, staircase_reversions] = deal(NaN([length(t2), n_perms, max_trials]));
for tidx = 1:length(t2)
    thresh = t2(tidx);
    [s_detect, s_amp, s_reversion, s_step_size] = deal(NaN([n_perms,max_trials]));
    jnd = thresh*0.4; % Convert to normal distribution Sigma
    sig = (1/norminv(.75))*jnd;
    parfor p = 1:n_perms        
        par_consecutive_criterion = consecutive_criterion; % Better if in parallel loop
        ground_truth_afc = normcdf(valid_stim_values, thresh, sig); % Create CDF/sigmoid

        step_size = init_step_size;
        new_direction = 0;
        noise_draws = rand([1,max_trials]);
        stim_amp_idx = randi(length(valid_stim_values),1);
        stim_amp = valid_stim_values(stim_amp_idx);
        current_direction = 0;
        consecutive_answers = [0,0];
        for t = 1:max_trials
            % Check if stim is valid
            if force_valid_stim_values
                [~, stim_idx] = min(abs(valid_stim_values - stim_amp));
                stim_draw = ground_truth_afc(stim_idx);
                s_amp(p,t) = valid_stim_values(stim_idx);
            else
                stim_draw = normcdf(stim_amp, thresh, sig);
                s_amp(p,t) = stim_amp;
            end
            s_step_size(p,t) = step_size;

            % Compare noise & stim draws + keep track of consecutive answers
            if stim_draw > noise_draws(t) || (num_afc > 1 && rand(1) < chance)
                s_detect(p,t) = 1;
                consecutive_answers(1) = consecutive_answers(1) + 1;
            else
                s_detect(p,t) = 0;
                consecutive_answers(2) = consecutive_answers(2) + 1;
            end

            % Only alter stimulus after cumulative hits
            if consecutive_answers(2) >= par_consecutive_criterion(2)
                new_direction = 1;
                stim_amp = stim_amp + step_size;
                consecutive_answers = [0,0];

            elseif consecutive_answers(1) >= par_consecutive_criterion(1)
                new_direction = -1;
                stim_amp = stim_amp - step_size;
                consecutive_answers = [0,0];
            end

            if t > 1 && current_direction ~= new_direction
                % Update step size
                if new_direction == -1
                    if strcmpi(step_method, 'fixed')
                        step_size = neg_step;
                    elseif strcmpi(step_method, 'convergent')
                        step_size = step_size * neg_step_coeff;
                    end
                elseif new_direction == 1
                    if strcmpi(step_method, 'fixed')
                        step_size = pos_step;
                    elseif strcmpi(step_method, 'convergent')
                        step_size = step_size * pos_step_coeff;
                    end
                end

                % Ensure step size does not runaway
                if step_size < min_step_size
                    step_size = min_step_size;
                elseif step_size > max_step_size
                    step_size = max_step_size;
                end

                current_direction = new_direction;
                s_reversion(p,t) = new_direction;
            end

            % Check amplitude is in stim range
            if force_valid_stim_values
                if stim_amp < min(valid_stim_values)
                   stim_amp = min(valid_stim_values);
                elseif stim_amp > max(valid_stim_values)
                   stim_amp = max(valid_stim_values);
                end
            end
        end
        send(pfwb_update, 0);
    end
    
    if example_plot && any(ismember(example_t2, tidx))
        AlphaLine(trial_vec, s_amp, t2_colors(tidx,:), 'ErrorType', 'STD', 'LineWidth', 2, 'Parent', f2_a1)
        target_stim = norminv(target_lvl, thresh, sig);
        scatter(max_trials*1.05, target_stim, 30, t2_colors(tidx,:), 'o', 'filled', 'Parent', f2_a1)
        predicted_p = (nthroot(.5, consecutive_criterion(1)/consecutive_criterion(2)) - chance) / chance;
        fprintf('Predicted convergence = %0.3f\n', predicted_p)
        converge_t = mean(s_amp(:,end-max_trials/2:end), 'all');
        p = normcdf(converge_t, thresh, sig);
        fprintf('Staircase converged at %1.1f %sA, p = %0.3f, p(correct) = %0.3f\n', converge_t, GetUnicodeChar('mu'), p, chance + chance*p)
    end
    
    staircase_detections(tidx, :,:) = s_detect;
    staircase_amplitudes(tidx, :,:) = s_amp;
    staircase_reversions(tidx, :,:) = s_reversion;
    
end
drawnow

%% Determine target value
%% Staircase accuracy as a function of JND
% Compute average of previous trials
max_trials_range = 10:250;
previous_trial_amps = 1:100;
[ta_tpred, ta_tpred_err] = deal(NaN([length(t2), length(max_trials_range), length(previous_trial_amps), n_perms]));
for mt_idx = 1:length(max_trials_range)
    mt = max_trials_range(mt_idx);
    for pt_idx = 1:length(previous_trial_amps)
        pt = previous_trial_amps(pt_idx);
        if pt <= mt
            ta_tpred(:,mt_idx,pt_idx,:) = mean(staircase_amplitudes(:,:,mt-pt+1:mt),3);
        end
    end
end

% Compute average of previous reversions - this is a touch more complicated as reversions
% are not consistent across permutations and so each must be addressed seperately
max_reversions_range = 2:50; % at chance every other trial will revert
previous_reversions = 1:20;
staircase_reversions_nan_check = ~isnan(staircase_reversions);
[ra_tpred, ra_tpred_err] = deal(NaN([length(t2), length(max_reversions_range), length(previous_reversions), n_perms]));
min_reversion = 100;
for t_idx = 1:length(t2)
    for p_idx = 1:n_perms
        reversion_idx = find(staircase_reversions_nan_check(t_idx,p_idx,:));
        for mr_idx = 1:length(max_reversions_range)
            mr = max_reversions_range(mr_idx);
            if length(reversion_idx) >= mr
                for pr_idx = 1:length(previous_reversions)
                    pr = previous_reversions(pr_idx);
                    if mr >= pr
                        amp_idx = reversion_idx(mr-pr+1:mr);
                        ra_tpred(t_idx, mr_idx, pr_idx, p_idx) = mean(staircase_amplitudes(t_idx, p_idx, amp_idx), 'all');
                    end
                end
            end
        end
    end
end

t_real = zeros([length(t2),1]);
% Adjust accuracy for each intensity
for i = 1:length(t2)
    thresh = t2(i);
    jnd = thresh*0.4; % Convert to normal distribution Sigma
    sig = (1/norminv(.75))*jnd;
    t_real(i) = norminv(target_lvl, thresh, sig);
    ta_tpred_err(i,:,:,:) = abs(ta_tpred(i,:,:,:) - t_real(i));
    ra_tpred_err(i,:,:,:) = abs(ra_tpred(i,:,:,:) - t_real(i));
end

%% 2a Plotting
figure;
    subplot(2,2,1);
    temp = squeeze(mean(ta_tpred_err, [1,4]));
    imAlpha = ones(size(temp));
    imAlpha(isnan(temp)) = 0;
    imagesc(max_trials_range, previous_trial_amps, temp', 'AlphaData', imAlpha')
    colorbar
    set(gca, 'Color', [0.95 0.95 0.95], 'YDir', 'reverse')

subplot(2,2,2);
    temp = squeeze(std(ta_tpred_err, 1,[1,4]));
    imAlpha = ones(size(temp));
    imAlpha(isnan(temp)) = 0;
    imagesc(max_trials_range, previous_trial_amps, temp', 'AlphaData', imAlpha')
    colorbar
    set(gca, 'Color', [0.95 0.95 0.95], 'YDir', 'reverse')

subplot(2,2,3);
    temp = squeeze(mean(ra_tpred_err, [1,4]));
    imAlpha = ones(size(temp));
    imAlpha(isnan(temp)) = 0;
    imagesc(max_reversions_range, previous_reversions, temp', 'AlphaData', imAlpha')
    colorbar
    set(gca, 'Color', [0.95 0.95 0.95], 'YDir', 'reverse')

subplot(2,2,4);
    temp = squeeze(std(ra_tpred_err, 1,[1,4]));
    imAlpha = ones(size(temp));
    imAlpha(isnan(temp)) = 0;
    imagesc(max_reversions_range, previous_reversions, temp', 'AlphaData', imAlpha')
    colorbar
    set(gca, 'Color', [0.95 0.95 0.95], 'YDir', 'reverse')
drawnow

%% 2a Plotting
t_idx = find(t2 == 50);
jnd = t2(t_idx)*0.4; % Convert to normal distribution Sigma
sig = (1/norminv(.75))*jnd;
p_idx1 = find(previous_reversions == 10);
p_idx2 = find(previous_trial_amps == 20);
target_stim = norminv(target_lvl, t2(t_idx), sig);
temp_ra_tpred = squeeze(ra_tpred(t_idx,:,p_idx1,:));
temp_ta_tpred = squeeze(ta_tpred(t_idx,:,p_idx2,:));
figure;
subplot(1,2,1); hold on
    plot([10,50],[target_stim,target_stim], 'Color', [.6 .6 .6], 'LineStyle', '--')
    AlphaLine(max_reversions_range(9:end), temp_ra_tpred(9:end,:), [.6 .6 .6], 'ErrorType', 'STD')
    xlim([10,50]); xlabel('# Reversions')
    ylim([40 70]); ylabel('Predicted Threshold'); yticks([40:10:70])

subplot(1,2,2); hold on
    plot([max_trials_range(1), max_trials_range(end)],[target_stim,target_stim], 'Color', [.6 .6 .6], 'LineStyle', '--')
    AlphaLine(max_trials_range, temp_ta_tpred, [.6 .6 .6],'ErrorType', 'STD')
    xlim([max_trials_range(1), max_trials_range(end)]); xlabel('# Trials')
    ylim([40 70]); yticks([40:10:70]); yticklabels({})

    
%% 3a Adaptive Limits - general
threshold = randi([10,90]); % Get a random threshold
jnd = threshold * (rand(1) * 0.2); % and get a random JND (with a max of t*0.4)
sigma = jnd2sig(jnd);

valid_stims = 2:2:98; % Get p(detection) for each stim value
ground_truth = getpdetect(valid_stims, threshold, sigma);

% Make detection array to keep track
max_repeats = 10; 
detection_array = NaN([length(valid_stims), max_repeats]);
clf; hold on
plot(valid_stims, ground_truth, 'LineStyle', '--', 'Color', [.6 .6 .6])
xlim([0 100])

starting_stim = randsample(valid_stims, 1); % Pick a random starting value
max_trials = 500;
min_stim_idx = 1;
max_stim_idx = length(valid_stims);

noise_draws = rand([1,max_trials]);
stim_amp = starting_stim;
for t = 1:max_trials
    % Work out p(detection) and where it is in the detection array
    stim_idx = find(valid_stims == stim_amp);
    stim_draw = ground_truth(stim_idx);
    dt_idx = find(isnan(detection_array(stim_idx,:)), 1, 'first');
    
    % Store detection in detection array
    if stim_draw > noise_draws(t)
        detection_array(stim_idx, dt_idx) = 1;
    else
        detection_array(stim_idx, dt_idx) = 0;
    end
    
    % Update graph
    if exist('s1', 'var')
        delete(s1)
    end
    running_dt = mean(detection_array,2, 'omitnan');
    running_trials_per_stim = sum(~isnan(detection_array),2);
    scatter_sizes = running_trials_per_stim ./ max_repeats * 100;
    nan_idx = ~isnan(running_dt);
    s1 = scatter(valid_stims(nan_idx), running_dt(nan_idx), scatter_sizes(nan_idx),...
        'MarkerFaceColor', [.6 .6 .6], 'MarkerEdgeColor', [.6 .6 .6], 'MarkerFaceAlpha', 0.3);
    
    % Determine what new stim level to use
    % At each stim level determine the odds of detecting the stim below, above
    [split_detection_rate, split_cumulative_trials] = deal(zeros([length(valid_stims),2]));
    for i = 1:length(split_detection_rate)
        % Number of trials and number of detections <= i
        ntb = sum(~isnan(detection_array(1:i,:)), 'all', 'omitnan');
        ndtb = sum(detection_array(1:i,:), 'all', 'omitnan');
        % Number of trials and number of detections >= i
        nta = sum(~isnan(detection_array(i:end,:)), 'all', 'omitnan');
        ndta = sum(detection_array(i:end,:), 'all', 'omitnan');
        % Compute split detection rate
        split_detection_rate(i,1) = ndtb / ntb;
        split_detection_rate(i,2) = ndta / nta;
        % keep track of trials
        split_cumulative_trials(i,1) = ntb;
        split_cumulative_trials(i,2) = nta;
    end
    % First update the upper and lower bounds - p(<x) < 0.95 where x = number of points 
    new_min_idx = find(split_detection_rate(:,1) < 0.05 & split_cumulative_trials(:,1) > max_repeats);
    if ~isempty(new_min_idx)
        min_stim_idx = new_min_idx(end) - 1;
    elseif isempty(new_min_idx) && split_detection_rate(min_stim_idx,1) > .1 && split_cumulative_trials(min_stim_idx,1) > max_repeats
        min_stim_idx = min_stim_idx - 1;
    end
    new_max_idx = find(split_detection_rate(:,2) > 0.95 & split_cumulative_trials(:,2) > max_repeats);
    if ~isempty(new_max_idx)
        max_stim_idx = new_max_idx(1) + 1;
    elseif isempty(new_max_idx) && split_detection_rate(max_stim_idx,1) > .9 && split_cumulative_trials(max_stim_idx,1) > max_repeats
        max_stim_idx = max_stim_idx + 1;
    end
    stim_in_range = range([min_stim_idx, max_stim_idx]);
    
    if exist('l1', 'var')
        delete(l1)
        delete(l2)
    end
    l1 = plot([valid_stims(min_stim_idx), valid_stims(min_stim_idx)], [0 1],...
        'LineStyle', ':', 'Color', [.6 .6 1]);
    l2 = plot([valid_stims(max_stim_idx), valid_stims(max_stim_idx)], [0 1],...
        'LineStyle', ':', 'Color', [1 .6 .6]);
    
    % Early stopping if all stims values in range have reached criteria (max_repeats)
    if all(~isnan(detection_array(min_stim_idx:max_stim_idx,max_repeats)))
        fprintf('Stopping at trial %d, all levels tested %d times\n', t, max_repeats)
        break
    end
    
    % Pick the valid stim with fewest repeats
    if detection_array(stim_idx, dt_idx)
        if stim_idx > min_stim_idx % Pick a lower stim
            [~, new_stim_idx] = min(running_trials_per_stim(min_stim_idx:stim_idx));
            new_stim_idx = new_stim_idx + min_stim_idx - 1;
        else % Pick any valid stim
            [~, new_stim_idx] = min(running_trials_per_stim(min_stim_idx:max_stim_idx));
            new_stim_idx = new_stim_idx + min_stim_idx - 1;
        end
            
    else
       if stim_idx < max_stim_idx % Pick a higher stim
            [~, new_stim_idx] = min(running_trials_per_stim(stim_idx:max_stim_idx));
            new_stim_idx = new_stim_idx + stim_idx - 1;
       else % Pick any valid stim
            [~, new_stim_idx] = min(running_trials_per_stim(min_stim_idx:max_stim_idx));
            new_stim_idx = new_stim_idx + min_stim_idx - 1;
        end
    end
    stim_amp = valid_stims(new_stim_idx);
    
    drawnow()
    pause(0.1)
end

% Fit sigmoid to data
idx2fit = find(running_trials_per_stim == max_repeats);
stims2fit = valid_stims(idx2fit);
detection2fit = mean(detection_array(idx2fit,:),2,'omitnan')';
% Check if we can add tails
if idx2fit(1) > valid_stims(1) && sum(running_trials_per_stim(1:idx2fit(1)-1)) >= max_repeats
    stims2fit = [valid_stims(idx2fit(1) - 1), stims2fit];
    detection2fit = [mean(detection_array(1:idx2fit(1) - 1,:),'all','omitnan'), detection2fit];
end
if idx2fit(end) < valid_stims(end) && sum(running_trials_per_stim(idx2fit(end)+1:end)) >= max_repeats
    stims2fit = [stims2fit, valid_stims(idx2fit(end) + 1)];
    detection2fit = [detection2fit, mean(detection_array(idx2fit(end) + 1:end,:),'all','omitnan')];
end
[~,b] = min(abs(detection2fit - 0.5));
[sig_c, ~] = lsqcurvefit(sigfun,[rand(1),stims2fit(b)], stims2fit, detection2fit, [0, 10],[1, 90], opts);
sig_y = sigfun(sig_c, valid_stims);
plot(valid_stims, sig_y, 'Color', rgb(126, 87, 194), 'LineStyle', '--')
fprintf('Actual DT = %d, Measured DT = %d\n', threshold, round(sig_c(2)))

%% 3b Adaptive Limits - 2 step
% 1. test 10:10:90 twice each to get rough limits
% 2. determine appropriate increments to test at
% 3. Run normal adaptive limits constrained to that range
threshold = randi([10,90]); % Get a random threshold
jnd = threshold * (rand(1) * 0.2); % and get a random JND (with a max of t*0.4)
sigma = jnd2sig(jnd);

clf; hold on
plot([0:100], getpdetect([0:100], threshold, sigma), 'LineStyle', ':', 'Color', [.6 .6 .6])
xlim([0 100])

% Coarse detection
coarse_valid_stims = 10:10:90; % Get p(detection) for each stim value
coarse_repeats = 2;
ground_truth = getpdetect(coarse_valid_stims, threshold, sigma);
stim_idx = repmat([1:length(coarse_valid_stims)], [1,2]);
stim_idx = stim_idx(randperm(length(stim_idx)));
coarse_detection_array = NaN([length(coarse_valid_stims), coarse_repeats]);
noise_draws = rand([1,length(stim_idx)]);
for t = 1:length(stim_idx)
    stim_draw = ground_truth(stim_idx(t));
    dt_idx = find(isnan(coarse_detection_array(stim_idx(t),:)), 1, 'first');
    if stim_draw > noise_draws(t)
        coarse_detection_array(stim_idx(t), dt_idx) = 1;
    else
        coarse_detection_array(stim_idx(t), dt_idx) = 0;
    end
    
    % Update graph
    if exist('s1', 'var')
        delete(s1)
    end
    
    running_dt = mean(coarse_detection_array,2, 'omitnan');
    running_trials_per_stim = sum(~isnan(coarse_detection_array),2);
    scatter_sizes = running_trials_per_stim ./ coarse_repeats * 100;
    nan_idx = ~isnan(running_dt);
    s1 = scatter(coarse_valid_stims(nan_idx), running_dt(nan_idx), scatter_sizes(nan_idx),...
        'MarkerFaceColor', [.6 .6 .6], 'MarkerEdgeColor', [.6 .6 .6], 'MarkerFaceAlpha', 0);
    drawnow()
    pause(0.1)
end

% Determine limits from course detection
[split_detection_rate, split_cumulative_trials] = deal(zeros([length(coarse_valid_stims),2]));
for i = 1:length(split_detection_rate)
    % Number of trials and number of detections <= i
    ntb = sum(~isnan(coarse_detection_array(1:i,:)), 'all', 'omitnan');
    ndtb = sum(coarse_detection_array(1:i,:), 'all', 'omitnan');
    % Number of trials and number of detections >= i
    nta = sum(~isnan(coarse_detection_array(i:end,:)), 'all', 'omitnan');
    ndta = sum(coarse_detection_array(i:end,:), 'all', 'omitnan');
    % Compute split detection rate
    split_detection_rate(i,1) = ndtb / ntb;
    split_detection_rate(i,2) = ndta / nta;
    % keep track of trials
    split_cumulative_trials(i,1) = ntb;
    split_cumulative_trials(i,2) = nta;
end

min_stim_idx = find(split_detection_rate(:,1) == 0, 1, 'last');
if isempty(min_stim_idx)
    min_stim_idx = 1;
end
max_stim_idx = find(split_detection_rate(:,2) == 1, 1, 'first');
if isempty(max_stim_idx)
    max_stim_idx = length(split_detection_rate);
end

if exist('l1', 'var')
        delete(l1)
        delete(l2)
end
l1 = plot([coarse_valid_stims(min_stim_idx), coarse_valid_stims(min_stim_idx)], [0 1],...
    'LineStyle', ':', 'Color', [.6 .6 1]);
l2 = plot([coarse_valid_stims(max_stim_idx), coarse_valid_stims(max_stim_idx)], [0 1],...
    'LineStyle', ':', 'Color', [1 .6 .6]);

% Fine detection
dt_range = range([coarse_valid_stims(min_stim_idx), coarse_valid_stims(max_stim_idx)]);
if dt_range == 10
    n_fine_points = 4;
else
    n_fine_points = 6;
end
% Work out what stims to use
fine_valid_stims = linspace(coarse_valid_stims(min_stim_idx), coarse_valid_stims(max_stim_idx), n_fine_points);
fine_valid_stims = 2 * round(fine_valid_stims / 2);

% Make detection array to keep track
max_fine_repeats = 15; 
detection_array = NaN([length(fine_valid_stims), max_fine_repeats]);

starting_stim = randsample(fine_valid_stims, 1); % Pick a random starting value
max_trials = 100;
min_stim_idx = 1;
max_stim_idx = length(fine_valid_stims);
ground_truth = getpdetect(fine_valid_stims, threshold, sigma);

noise_draws = rand([1,max_trials]);
stim_amp = starting_stim;
for t2 = 1:max_trials
    % Work out p(detection) and where it is in the detection array
    stim_idx = find(fine_valid_stims == stim_amp);
    stim_draw = ground_truth(stim_idx);
    dt_idx = find(isnan(detection_array(stim_idx,:)), 1, 'first');
    
    % Store detection in detection array
    if stim_draw > noise_draws(t2)
        detection_array(stim_idx, dt_idx) = 1;
    else
        detection_array(stim_idx, dt_idx) = 0;
    end
    
    % Update graph
    if exist('s2', 'var')
        delete(s2)
    end
    running_dt = mean(detection_array,2, 'omitnan');
    running_trials_per_stim = sum(~isnan(detection_array),2);
    scatter_sizes = running_trials_per_stim ./ max_fine_repeats * 100;
    nan_idx = ~isnan(running_dt);
    s2 = scatter(fine_valid_stims(nan_idx), running_dt(nan_idx), scatter_sizes(nan_idx),...
        'MarkerFaceColor', [.6 .6 .6], 'MarkerEdgeColor', [.6 .6 .6], 'MarkerFaceAlpha', .6);
    
    % Determine what new stim level to use
    % At each stim level determine the odds of detecting the stim below, above
    [split_detection_rate, split_cumulative_trials] = deal(zeros([length(fine_valid_stims),2]));
    for i = 1:length(split_detection_rate)
        % Number of trials and number of detections <= i
        ntb = sum(~isnan(detection_array(1:i,:)), 'all', 'omitnan');
        ndtb = sum(detection_array(1:i,:), 'all', 'omitnan');
        % Number of trials and number of detections >= i
        nta = sum(~isnan(detection_array(i:end,:)), 'all', 'omitnan');
        ndta = sum(detection_array(i:end,:), 'all', 'omitnan');
        % Compute split detection rate
        split_detection_rate(i,1) = ndtb / ntb;
        split_detection_rate(i,2) = ndta / nta;
        % keep track of trials
        split_cumulative_trials(i,1) = ntb;
        split_cumulative_trials(i,2) = nta;
    end
    % First update the upper and lower bounds - p(<x) < 0.95 where x = number of points 
    new_min_idx = find(split_detection_rate(:,1) < 0.05 & split_cumulative_trials(:,1) > max_fine_repeats);
    if ~isempty(new_min_idx)
        min_stim_idx = new_min_idx(end) - 1;
    elseif isempty(new_min_idx) && split_detection_rate(min_stim_idx,1) > .1 && split_cumulative_trials(min_stim_idx,1) > max_fine_repeats
        min_stim_idx = min_stim_idx - 1;
    end
    new_max_idx = find(split_detection_rate(:,2) > 0.95 & split_cumulative_trials(:,2) > max_fine_repeats);
    if ~isempty(new_max_idx)
        max_stim_idx = new_max_idx(1) + 1;
    elseif isempty(new_max_idx) && split_detection_rate(max_stim_idx,1) > .9 && split_cumulative_trials(max_stim_idx,1) > max_fine_repeats
        max_stim_idx = max_stim_idx + 1;
    end
    stim_in_range = range([min_stim_idx, max_stim_idx]);
    
    
% Early stopping if all stims values in range have reached criteria (max_fine_repeats)
if all(~isnan(detection_array(min_stim_idx:max_stim_idx,max_fine_repeats)))
    fprintf('Stopping at trial %d, all levels tested %d times\n', t+t2, max_fine_repeats)
    break
end
    
    % Pick the valid stim with fewest repeats
    if detection_array(stim_idx, dt_idx)
        if stim_idx > min_stim_idx % Pick a lower stim
            [~, new_stim_idx] = min(running_trials_per_stim(min_stim_idx:stim_idx));
            new_stim_idx = new_stim_idx + min_stim_idx - 1;
        else % Pick any valid stim
            [~, new_stim_idx] = min(running_trials_per_stim(min_stim_idx:max_stim_idx));
            new_stim_idx = new_stim_idx + min_stim_idx - 1;
        end
            
    else
       if stim_idx < max_stim_idx % Pick a higher stim
            [~, new_stim_idx] = min(running_trials_per_stim(stim_idx:max_stim_idx));
            new_stim_idx = new_stim_idx + stim_idx - 1;
       else % Pick any valid stim
            [~, new_stim_idx] = min(running_trials_per_stim(min_stim_idx:max_stim_idx));
            new_stim_idx = new_stim_idx + min_stim_idx - 1;
        end
    end
    stim_amp = fine_valid_stims(new_stim_idx);
    
    drawnow()
    pause(0.1)
end

% Fit sigmoid to data
idx2fit = find(running_trials_per_stim == max_fine_repeats);
stims2fit = fine_valid_stims(idx2fit);
detection2fit = mean(detection_array(idx2fit,:),2,'omitnan')';
% Check if we can add tails
if idx2fit(1) > fine_valid_stims(1) && sum(running_trials_per_stim(1:idx2fit(1)-1)) >= max_fine_repeats
    stims2fit = [fine_valid_stims(idx2fit(1) - 1), stims2fit];
    detection2fit = [mean(detection_array(1:idx2fit(1) - 1,:),'all','omitnan'), detection2fit];
end
if idx2fit(end) < fine_valid_stims(end) && sum(running_trials_per_stim(idx2fit(end)+1:end)) >= max_fine_repeats
    stims2fit = [stims2fit, fine_valid_stims(idx2fit(end) + 1)];
    detection2fit = [detection2fit, mean(detection_array(idx2fit(end) + 1:end,:),'all','omitnan')];
end
[~,b] = min(abs(detection2fit - 0.5));
[sig_c, resnorm] = lsqcurvefit(sigfun,[rand(1),stims2fit(b)], stims2fit, detection2fit, [0, 10],[1, 90], opts);
plot([0:100], sigfun(sig_c, [0:100]), 'Color', rgb(126, 87, 194), 'LineStyle', '--')
fprintf('Actual DT = %d, Measured DT = %d\n', threshold, round(sig_c(2)))

%%
% Measure total time required for all tasks to get similar detection threshold performance
% Measure performance per unit time for QUEST, PEST, MLE


%% Waitbar incrementer
function wbincrement(wb)
    ud = wb.UserData;
    ud(1) = ud(1) + 1;
    waitbar(ud(1) / ud(2), wb); % sprintf('Simulation %d of %d', ud(1), ud(2))
    wb.UserData = ud;
end





