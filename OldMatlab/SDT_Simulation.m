%%% Simulating detection thresholds
intensity = [1:0.1:100];

% Let's assume that the threshold is 25 ma 
% The JND will be 1/5 of threshold
% The 25th/75th percentile values will be 15 and 25 respectively

% Sigmoid function for estimation
sigfun = @(c,x) 1./ (1 + c(1).^(x-c(2)));
opts = optimset('Display','off');

% Create CDF for truth
t = 50;
jnd = t*0.4;
z = norminv([0.75]);
sigma = z * jnd * 2;
ground_truth = normcdf(intensity, t, sigma);
stims = [t-(5*4):5:t+(5*4)]; % 9 stimuli centered on t

%% Absolute detection - detection only (threshold theory)
% Simulate responses to each stimulus intensity
n_repeats = 10000; % Number of stimuli at each intensity

[detections, p_detect] = deal(zeros([length(stims), n_repeats]));
draw = rand([length(stims), n_repeats]);

for s = 1:length(stims)
    [~, intensity_idx] = min(abs(intensity-stims(s)));
    p_detect(s,:) = ground_truth(intensity_idx);
end

detections = p_detect > draw;
detections_10 = mean(detections(:,1:10),2);
detections_50 = mean(detections(:,1:50),2);
detections_100 = mean(detections(:,1:100),2);

% MSE
max_repeats = 100;
n_folds = n_repeats/max_repeats;
y_truth = p_detect(:,1);
[repeat_mse, sig_mse] = deal(zeros([n_folds, max_repeats]));
sig_t = zeros([n_folds, max_repeats]);
shuffle_idx = randperm(n_repeats);
shuffle_idx_fold = reshape(shuffle_idx, [n_folds, floor(n_repeats/n_folds)]);

for i = 1:max_repeats
    for f = 1:n_folds
        temp_detections = detections(:,shuffle_idx_fold(f,:));
        temp_detections = mean(temp_detections(:,1:i),2);
        
        repeat_mse(f,i) = mean(sum((temp_detections - y_truth).^2));

        sig_c = lsqcurvefit(sigfun,[1,25], stims, temp_detections', [],[],opts);
        sig_y = sigfun(sig_c, stims);
        sig_mse(f,i) = mean(sum((sig_y - y_truth').^2));
        sig_t(f,i) = sig_c(2);
    end
end

figure;
subplot(1,2,1); hold on
% True distribution
plot(intensity, ground_truth, 'Color', [0.6 0.6 0.6], 'LineStyle', '--')
plot([t,t,0], [0, 0.5, 0.5], 'Color', [0.6 0.6 0.6], 'LineStyle', ':')
% Observed distribution
scatter(stims, detections_10, 30, 'MarkerEdgeColor', [103, 58, 183]./255) 
scatter(stims, detections_50, 30, 'MarkerEdgeColor', [255, 152, 0]./255)
scatter(stims, detections_100, 30, 'MarkerEdgeColor', [0, 150, 136]./255)

leg_text = [{['\color[rgb]{', num2str([103, 58, 183]./255), '}',  '10']};...
        {['\color[rgb]{', num2str([255, 152, 0]./255), '}',  '50']};...
        {['\color[rgb]{', num2str([0, 150, 136]./255), '}',  '100']}];
    
text(5, 0.975, leg_text, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left')

ylabel('p(Detection)')
xlabel('Amplitude')

subplot(1,2,2); hold on
plot([1:max_repeats], mean(repeat_mse,1), 'Color', [63, 81, 181]./255)
plot([1:max_repeats], prctile(repeat_mse,25,1), 'Color', [63, 81, 181]./255, 'LineStyle', '--')
plot([1:max_repeats], prctile(repeat_mse,75,1), 'Color', [63, 81, 181]./255, 'LineStyle', '--')
plot([1:max_repeats], mean(sig_mse,1), 'Color', [211, 47, 47]./255)
plot([1:max_repeats], prctile(sig_mse,75,1), 'Color', [211, 47, 47]./255, 'LineStyle', '--')
plot([1:max_repeats], prctile(sig_mse,25,1), 'Color', [211, 47, 47]./255, 'LineStyle', '--')

set(gca, 'XScale', 'log')
xlabel('# Stimuli/Intensity')
ylabel('MSE')

leg_text = [{['\color[rgb]{', num2str([63, 81, 181]./255), '}',  'Raw']};...
            {['\color[rgb]{', num2str([211, 47, 47]./255), '}',  'Sigmoid']}];
    
    
%% Difference thresholds - SDT
d_prime = [-9:0.1:9];
n_reps = 1000;
detections = zeros([length(d_prime), n_reps]);

for d = 1:length(d_prime)
    noise_mean = t - (sigma/2*d_prime(d));
    n_draw = normrnd(noise_mean,sigma, [1,n_reps]);
    ns_draw = normrnd(t,sigma, [1,n_reps]);
    detections(d,:) = ns_draw > n_draw;
end

d_prime_intensities = t + (sigma/2.*d_prime);
figure; hold on
plot([t,t], [0 .5], 'LineStyle', '--', 'Color', [.6 .6 .6])
plot([0,t], [.5 .5], 'LineStyle', '--', 'Color', [.6 .6 .6])
plot(d_prime_intensities, mean(detections,2), 'Color', lines(1))
xlabel('Intensity'); ylabel('p(below) - p(above)')


%% Absolute detection - forced choice paradigm (using SDT model)
% (p-0.5)*2 ?
d_prime = [-9:.1:9];
n_reps = 10000;

% The 'noise' distribution will always be the -8 standard deviations from
% the detection threshold. The mean of the noise distribution is thus
% dependent on the absolute threshold (t) and the variance of the noise
% distribution (which is also the JND at threshold)
% Therefor:

absolute_threshold = 50;
sigma = absolute_threshold * 0.2;
noise_mean = absolute_threshold + (sigma/2*-4);

% On each trial there will be a draw from the noise and a draw from the
% noise plus signal. The 'subject' will pick the value with the higher
% draw. Because there is no bias we do not need to counterbalance the
% stimulus/catch order.

detections = zeros([length(d_prime), n_reps]);
for d = 1:length(d_prime)
    noise_draw = normrnd(noise_mean, sigma, [1,n_reps]);
    
    noise_plus_signal_mean = absolute_threshold + (sigma/2*d_prime(d));
    if noise_plus_signal_mean < noise_mean
        noise_plus_signal_mean = noise_mean;
    end
    noise_plus_signal_draw = normrnd(noise_plus_signal_mean, sigma, [1,n_reps]);
    
    for r = 1:n_reps
        % Decision logic
        % The subject is only 'guaranteed' to know in which interval the
        % stimulus is if the draw is above threshold in the NPS
        % draw but not the N draw
        if noise_plus_signal_draw(r) > absolute_threshold & noise_draw(r) < absolute_threshold
            detections(d,r) = 1;
            
        % In both other cases (both above threshold or both below
        % threshold) then the decision is random
        elseif noise_plus_signal_draw(r) > absolute_threshold & noise_draw(r) > absolute_threshold & noise_plus_signal_draw(r) > noise_draw(r)
            if rand(1) > 0.5
                detections(d,r) = 1;
            end
        elseif noise_plus_signal_draw(r) < absolute_threshold & noise_draw(r) < absolute_threshold
            if rand(1) > 0.5
                detections(d,r) = 1;
            end
        end
    end
    
end

d_prime_intensities = absolute_threshold + (sigma/2.*d_prime);

temp = mean(detections,2);
temp_adj = (temp-0.5)/(1-0.5);
figure; hold on
plot([t,t], [.5 .75], 'LineStyle', '--', 'Color', [.6 .6 .6])
plot([0,t], [.75 .75], 'LineStyle', '--', 'Color', [.6 .6 .6])
plot(d_prime_intensities, temp, 'Color', lines(1))
ylim([.45 1]); ylabel('p(Detected)'); xlabel('Intensity')

%% Simple staircase - Simulation
threshold = 50;
jnd = threshold*0.2;
z = norminv([0.75]);
sigma = z * jnd * 2;

valid_stim_values = 2:2:98;
ground_truth = normcdf(valid_stim_values, threshold, sigma);

init_step_size = 10;%2*round((init_amp/2)/2);
max_trials = 100;
num_simulations = 1000;

step_size = init_step_size;
min_step_size = 2;
detect_vec = NaN([num_simulations,max_trials]);
amp_vec = NaN([num_simulations,max_trials]);

%stopping_criteria = NumTrials, NumReversions, AmpStability, ReversionsAfterMinStep, TrialsAfterMinStep
for s = 1:num_simulations
    noise_draws = rand([1,max_trials]);
    stim_amp_idx = randi(length(valid_stim_values),1);
    stim_amp = valid_stim_values(stim_amp_idx);
    current_direction = 0;
    for t = 1:max_trials
        % Draw noise and stim
        stim_idx = find(valid_stim_values == stim_amp);
        stim_draw = ground_truth(stim_idx);
        amp_vec(s,t) = stim_amp;

        if stim_draw > noise_draws(t)
            detect_vec(s,t) = 1;
            new_direction = 1;
            stim_amp = stim_amp - step_size;

        elseif noise_draws(t) > stim_draw
            detect_vec(s,t) = 0;
            new_direction = -1;
            stim_amp = stim_amp + step_size;
        end
        
        if current_direction == 0
            % Don't update step size on first trial
            current_direction = new_direction;
            
        elseif current_direction ~= new_direction
            % Update step size
            if step_size > min_step_size
                step_size = step_size - 2*round((step_size/2)/2);
            end
            
            % Ensure step size does not fall below minimum
            if step_size < min_step_size
                step_size = min_step_size;
            end
            
            current_direction = new_direction;
        end
        
        % Check amplitude is in stim range
        if stim_amp < min(valid_stim_values)
           stim_amp = min(valid_stim_values);
        elseif stim_amp > max(valid_stim_values)
           stim_amp = max(valid_stim_values);
        end
    end
end

trial_vec = [1:max_trials];
figure;
subplot(2,4,[1,2,5,6]); 
AlphaLine(trial_vec,amp_vec, [.6 .6 .6], 'ErrorType', 'STD', 'LineWidth', 2)
scatter(max_trials +2, threshold, 'o', 'k', 'filled')
ylim([0 100]); ylabel('Stimulus'); xlabel('Trial')
xlim([1, 102])

for i = [3,4,7,8]
    subplot(2,4,i); hold on
    % Draw a random trial
    t_idx = randi(num_simulations);
    temp_detect_vec = detect_vec(t_idx,:);
    temp_amp_vec = amp_vec(t_idx,:);
    plot([1,max_trials], [threshold threshold], 'LineStyle', '--', 'Color', [.6 .6 .6])
    scatter(trial_vec(temp_detect_vec == 1), temp_amp_vec(temp_detect_vec == 1),  'o', 'g', 'filled')
    scatter(trial_vec(temp_detect_vec == 0), temp_amp_vec(temp_detect_vec == 0),  'o', 'r', 'filled')
    ylim([0 100]); yticks([]); xlim([0 100])
    xticks([])
end

% Simple staircase - Analysis
% Average of previous trials
max_trials_range = [19:100];
previous_trial_amps = [1:100];
thresh_pred = NaN([length(max_trials_range), length(previous_trial_amps), num_simulations]);
for mt_idx = 1:length(max_trials_range)
    mt = max_trials_range(mt_idx);
    for pt_idx = 1:length(previous_trial_amps)
        pt = previous_trial_amps(pt_idx);
        try
            thresh_pred(mt_idx,pt_idx,:) = mean(amp_vec(:,mt-pt+1:mt),2);
        end
    end
end
amp_error = mean(sqrt((thresh_pred - threshold).^2),3);
amp_variance = std(thresh_pred,1,3);
%%
imAlpha = ones(size(amp_error));
imAlpha(isnan(amp_error)) = 0;

amp_error_thresholded = amp_error;
amp_error_thresholded(amp_error_thresholded > prctile(amp_error(:),75)) = prctile(amp_error(:),75);
figure; 
subplot(1,2,1);
contourf(max_trials_range, previous_trial_amps, amp_error_thresholded', 20)%, 'AlphaData', imAlpha')
ylabel('Average of previous N trials');
xlabel('Trials in set')
caxis([prctile(amp_error(:),5), prctile(amp_error(:),75)]); colorbar
title('Detection Threshold Error')
set(gca, 'Color', [0.95 0.95 0.95], 'YDir', 'reverse')

subplot(1,2,2); 
imagesc(max_trials_range, previous_trial_amps, amp_variance', 'AlphaData', imAlpha')
title('Detection Threshold Standard Deviation')
xlabel('Trials in set'); yticklabels({})
caxis([prctile(amp_variance(:),5), prctile(amp_variance(:),75)]); colorbar
colorbar
set(gca, 'Color', [0.95 0.95 0.95], 'YDir', 'reverse')

set(gcf, 'Units', 'pixels', 'Position', os_fig_position([40 20]));


%%% Things to think about:
% 1. Different stopping criteria
% 2. Different search methods
% 3. Skip responses effect on time to accuracy


















