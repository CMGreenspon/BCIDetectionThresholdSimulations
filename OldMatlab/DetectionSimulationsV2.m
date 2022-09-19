%%% Detection Threshold Simulations & Analysis v2
% 1. Method of constant effects on threshold variance:
% 1a. Effect of trial count (determine reasonable number -- 10 per condition)
% 1c. Effect of JND sample density/number of points

% 2. Staircases
% 2a. Simple up/down threshold variance (trials/reversions)
% 2b. Effect of JND on threshold variance for given stopping point
% 2c. Compare different algorithms? (QUEST, PEST)

% 3. Adaptive constant
% 3a. Coarse constant: reps per intensity
% 3b. Fine constant: reps per intensity
% 3c. 2-step

% 4. Variance per trial summary
% Expect the constant to never really asymptote while the staircases may get
% closer quicker but asymptote higher

% Supp 1.
% S1a. Effect of number of trials on JND estimates
% S1b. Effect of num AFC

% Supp 2. 
% S2a. Error per threshold (range limitations)
% S2b. Random versus correct start

%%% Story in the paper
% A common experiment in human BCI is repeated estimates of an electrodes
% detection threshold over time. This helps us determine how sensitive an
% electrode is on a given day but also gives us insight into how stable or
% reliable an evoked sensation is likely to be from day to day.

% Adaptive or tracking algorithms are most commonly used when measuring 
% detection thresholds because they allow you to rapidly converge on the
% correct value, especially if you do not have a strong prior about what
% the detection threshold should be. In longitudinal experiments, however,
% this is not the case. Indeed, if the sensitivity of an electrode is
% stable then repeated measurements should give similar values. The option
% to use previous information to inform the next detection experiment may
% alter the efficiency metrics when repeatedly performing detection
% threshold experiments. We sought to determine the most efficient and
% accurate way to estimate detection thresholds within the framework of
% typical BCI experiments.

% Results:
% 1. Simulated 3/1 vs classical/constrained constant approaches.
% 1a. Staircases are indeed drastically more efficient than classical
% constant.
% 1b. However, if using prior information to inform the stimuli used for a
% constant approach it is actually equivalent or better than the staircase.
% 1c. To make this fair we conpared with staircases that also start at
% approximately the correct location and found that this doesn't improve
% performance

% 2. Comparing across range of DTs and JNDs at 60 Trials


%% Setup
SetFont('Arial', 16)
%set(0,'DefaultFigureWindowStyle','docked')
export_folder = 'C:\Users\somlab\Desktop\ExportedImages';

% Psychometric helper functions
sigfun = @(c,x) 1./ (1 + exp(-c(1).*(x-c(2)))); % c(1) = rate of change, c(2) = x-offset
invsig = @(c,y) (log(1/y - 1)/-c(1)) + c(2);
mae = @(x, xbar, dim) mean(abs(x-xbar), dim, 'omitnan');
rmse = @(x, xbar, dim) mean(sqrt((abs(x-xbar).^2)), dim, 'omitnan');
jnd2sigma = @(j) (1/norminv(.75))*j;
sigma2k = @(sigma) 1.7 / sigma;
sigc2jnd = @(sig_c) (invsig(sig_c, .75) - invsig(sig_c, .25)) / 2;
getpdetect = @(stim, threshold, sigma) normcdf(stim, threshold, sigma);

% add to struct for easier handling
psych.sigfun = sigfun;
psych.invsig = invsig;
psych.jnd2sigma = jnd2sigma;
psych.sigc2jnd = sigc2jnd;
psych.sigma2k = sigma2k;
psych.getpdetect = getpdetect;

% BCI stims
valid_stims = [2:2:98];

constant_col1 = rgb(171, 71, 188);
constant_col2 = rgb(123, 31, 162);
staircase_col = rgb(255, 112, 67);

opts = optimset('Display','off'); % Disable reporting for lsqcurvefit

%% 0. Intro figure
t = 50; j = 5; s = jnd2sigma(j);
st_p = GetStaircaseTarget(2, [3,1]);
dt_sc = norminv(st_p, t, s);
    
fig1 = figure('Name', 'Intro Figure');
f1_sp1 = subplot(1,3,1, 'Parent', fig1); hold on
    plot([0:.1:100], getpdetect([0:.1:100], t, s), 'Color', [.6 .6 .6])
    plot([t t 0], [0 .5 .5],  'Color', [.6 .6 .6], 'LineStyle', '--')
    plot([0, norminv(.25, t, s), norminv(.25, t, s)],...
        [getpdetect(norminv(.25, t, s), t, s), getpdetect(norminv(.25, t, s), t, s), 0],...
        'Color', [.6 .6 .6], 'LineStyle', ':')
    plot([0, norminv(.75, t, s), norminv(.75, t, s)],...
        [getpdetect(norminv(.75, t, s), t, s), getpdetect(norminv(.75, t, s), t, s), 0],...
        'Color', [.6 .6 .6], 'LineStyle', ':')
    text(100, .05, {sprintf('DT_{%0.0f} = %0.0f %sA', st_p*100, dt_sc, GetUnicodeChar('mu'));...
                    sprintf('DT_{50} = %d %sA', t, GetUnicodeChar('mu'));...
                    sprintf('JND = %d %sA', j, GetUnicodeChar('mu'))},...
                    'HorizontalAlignment', 'right', 'VerticalAlignment',...
                    'bottom', 'FontSize', 9, 'Color', [.6 .6 .6])
         
    xlabel(sprintf('Stimulus Intensity (%sA)', GetUnicodeChar('mu'))); xticks([0:25:100]);
    ylabel('p(Detect)'); yticks([0:.25:1])
    
f1_sp2 = subplot(1,3,2, 'Parent', fig1); hold on
    [target_stims, p_detect] = GetConstantTargets(getpdetect(valid_stims, t, s),...
        valid_stims, 'Unconstrained', 12);
    % For plotting examples only
    stim_draw = rand([length(target_stims), 10]);
    afc_draw = rand(size(stim_draw));
    % Deterimine detection 
    detections = p_detect > stim_draw | afc_draw > 1 - .5;
    temp_d = mean(detections,2);
    temp_d = (temp_d-.5) * (1/.5);
    temp_d(temp_d<0) = 0;
    plot([0:.1:100], getpdetect([0:.1:100], t, s), 'Color', [.6 .6 .6], 'LineStyle', '--')
    scatter(target_stims, temp_d, 50, 'MarkerFaceColor', constant_col1, ...
        'MarkerEdgeColor', constant_col1, 'MarkerFaceAlpha', 0.1)
    [sig_c, ~] = lsqcurvefit(sigfun,[rand(1),randi([10,90])], target_stims, temp_d, [0,1], [1 99], opts);
     plot([0:.1:100], sigfun(sig_c, [0:.1:100]), 'Color', constant_col1)
     
     text(100, .05, {sprintf('DT_{50} = %0.0f %sA', sig_c(2), GetUnicodeChar('mu'));...
         sprintf('JND = %0.0f %sA', sigc2jnd(sig_c), GetUnicodeChar('mu'))}, 'HorizontalAlignment', 'right',...
         'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', constant_col1)
    
    xlabel(sprintf('Stimulus Intensity (%sA)', GetUnicodeChar('mu'))); xticks([0:25:100])
    yticks([0:.25:1]); yticklabels({''}); 
    
f1_sp3 = subplot(1,3,3, 'Parent', fig1); hold on
    [amplitude_history, ~, reversion_idx] = StaircaseSimulation(t, s, valid_stims, 2,...
        [3,1], 'Convergent', 120, 1, 0, 20, true);
    plot([1,length(amplitude_history)], [dt_sc, dt_sc], 'Color', [.6 .6 .6], 'LineStyle', '--')
    plot([1:length(amplitude_history)], amplitude_history, 'Color', staircase_col)
    [pred_t, ~, ~, ~] = EstimateStaircaseThreshold(amplitude_history, reversion_idx,...
    'Reversion', 5);
    
    leg_text = ColorText({sprintf('DT_{%0.0f} = %0.0f', st_p*100, pred_t(end)),...
                          'JND = ?'}, [staircase_col; staircase_col]);
    text(120, 5, leg_text, 'HorizontalAlignment', 'right',...
         'VerticalAlignment', 'bottom', 'FontSize', 9)
    
    ylim([0 100]); yticks([0:25:100])
    ylabel(sprintf('Stimulus Intensity (%sA)', GetUnicodeChar('mu'))); 
    xlabel('# Trials'); xticks([0:60:120])

set(fig1, 'Units', 'pixels', 'Position', OSScreenSize([9.32, 3.74], 'in', 1));
%ffname = fullfile(export_folder, 'DT_IntroFigure');
%print(gcf,ffname, '-dpng', '-r600')

%% 1. Method of Constant Stimulus (Unconstrained) vs Transformed Staircase
t = 50; j = 5; s = jnd2sigma(j);

n_perms = 1e3; % (mr = 20) 1e2 = 12s, 1e3 = 2 mins, 1e4 = 25 mins, 1e5
max_repeats = 100; % Number of stimuli at each intensity
num_points = 8;
num_afc = 2;
n_sig_attempts = 1;
target_method = 'Unconstrained'; % Constrained/Unconstrained
bound_logit = 0;

% Create CDF/sigmoid
ground_truth = getpdetect(valid_stims, t, s);
% Get appropriate stims
[target_stims, p_detect] = GetConstantTargets(ground_truth, valid_stims, target_method, num_points);
tic
[wb, pfwb_update]  = ParforWaitbar('Constant simulation', max_repeats);
[t_repeats, jnd_repeats] = deal(zeros([n_perms, max_repeats]));
parfor r = 1:max_repeats
    [t_repeats(:,r), jnd_repeats(:,r)] = ConstantSimulation(psych, target_stims, p_detect, r, n_perms,...
        1/num_afc, n_sig_attempts, bound_logit, t, j, 0);
    send(pfwb_update, 0);
end
toc
[amplitude_history, ~, reversion_idx] = StaircaseSimulation(t, s, valid_stims, 2,...
        [3,1], 'Convergent', max_repeats*num_points*2, n_perms, 0, 20, true);
[pred_t, trial_count, pred_t_sorted, trial_count_x] = EstimateStaircaseThreshold(amplitude_history, reversion_idx,...
'Reversion', 5);

%% __ Plotting
fig2 = figure('Name', 'Unconstrained vs Staircase');
f2_sp1 = subplot(1,3,1, 'Parent', fig2); hold on
    plot([num_points,max_repeats *num_points], [t, t], 'Color', [0.6 0.6 0.6],...
            'LineStyle', '--', 'Parent', f2_sp1)
    AlphaLine([num_points:num_points:max_repeats*num_points], t_repeats', constant_col1,...
        'ErrorType', 'STD', 'Parent', f2_sp1, 'EdgeAlpha', 0)
    ylabel('Threshold Estimate'); 
    xticks([12,60,240]); xlabel('# Trials'); 
    set(gca, 'XScale', 'log')
    ylim([0 100]); xlim([10 max_repeats *num_points]); yticks([0:25:100])
    
f2_sp2 = subplot(1,3,2, 'Parent', fig2); hold on
    target_t = norminv(GetStaircaseTarget(2, [3,1]), t, s);
    plot([trial_count_x(2),max_repeats *num_points], [target_t, target_t], 'Color', [0.6 0.6 0.6],...
            'LineStyle', '--', 'Parent', f2_sp2)
    AlphaLine(trial_count_x, pred_t_sorted, staircase_col, 'ErrorType', 'STD', 'Parent', f2_sp2, 'EdgeAlpha', 0)
    ylim([0 100]); xlim([0 max_repeats *num_points]); yticks([0:25:100]); yticklabels({})
    set(gca, 'XScale', 'log');xticks([12,60,240]); xlabel('# Trials'); 

f2_sp3 = subplot(1,3,3, 'Parent', fig2); hold on
    plot([1:max_repeats]*num_points, rmse(t_repeats,t,1)/s, 'Color', constant_col1, 'Parent', f2_sp3, 'LineWidth', 2)
    plot(trial_count_x, rmse(pred_t_sorted, target_t, 1)/s, 'Color', staircase_col, 'Parent', f2_sp3, 'LineWidth', 2)
    set(f2_sp3, 'XScale', 'log', 'YScale', 'log', 'XLim', [10 max_repeats *num_points])
    xlabel('# Trials'); xticks([12,60,240]);
    ylabel('Error (Z)'); yticks([.1 .25 .5 1 2 4]); ylim([.1 6.5])

set(fig2, 'Units', 'pixels', 'Position', OSScreenSize([9.32, 3.74], 'in', 1));
%ffname = fullfile(export_folder, 'DT_UnconstrainedComparison');
%print(fig2,ffname, '-dpng', '-r600')

%% S1ai. JND Coverage
t = 50; j = 10; s = jnd2sigma(j);
ground_truth = getpdetect(valid_stims, t, s);

n_perms = 1e2; % Number of times to repeat each simulation
n_repeats = 1e1; % number of trials per stimulus level
n_sig_attempts = 5;

point_range = [2:20];
point_density = [0.1:.1:1];

[constant_t_j, constant_t_j_bounded] = deal(NaN([length(point_range), length(point_density), n_perms]));
[wb, pfwb_update]  = ParforWaitbar('S1a. Stimulus coverage', length(point_range)*length(point_density));
for r = 1:length(point_range)
    parfor d = 1:length(point_density)
        % Determine stimulus distribution
        lb = .5 - (point_density(d)/2);
        ub = .5 + (point_density(d)/2);
        target_p = linspace(lb,ub,point_range(r))'; %#ok<*PFBNS>
    
        % Get uniformly distributed (in probability space) stim levels
        [t_stims, p_detect] = deal(zeros(size(target_p)));
        for n = 1:point_range(r)
            [a,b] = min(abs(ground_truth-target_p(n)));
            t_stims(n,1) = valid_stims(b);
            p_detect(n,1) = ground_truth(b);
        end
        % Remove overlapping
        [~,u_idx] = unique(t_stims);
        t_stims = t_stims(u_idx);
        p_detect = p_detect(u_idx);
        
        [pred_t, ~] = ConstantSimulation(psych, t_stims, p_detect, n_repeats, n_perms, 0.5,...
                                                n_sig_attempts, false, t, j, 0);
        constant_t_j(r,d,:) = pred_t;
        
        [pred_t, ~] = ConstantSimulation(psych, t_stims, p_detect, n_repeats, n_perms, 0.5,...
                                                n_sig_attempts, true, t, j, t);
        constant_t_j_bounded(r,d,:) = pred_t;
        send(pfwb_update, 0);
    end
end

%% S1ai. Repeats
% Take 60% of the sigmoid and vary number of levels vs number of repeats (bounded only)
pr = 0.6;
lb = .5 - (pr/2);
ub = .5 + (pr/2);

n_repeats = [1:20];
constant_t_j_bounded_reps = NaN([length(n_repeats), length(point_range), n_perms]);
[wb, pfwb_update]  = ParforWaitbar('S1a. Stimulus coverage', length(n_repeats)*length(point_range));
for r = 1:length(n_repeats)
    parfor p = 1:length(point_range)
        target_p = linspace(lb,ub,point_range(p))'; %#ok<*PFBNS>

        % Get uniformly distributed (in probability space) stim levels
        [t_stims, p_detect] = deal(zeros(size(target_p)));
        for n = 1:point_range(p)
            [a,b] = min(abs(ground_truth-target_p(n)));
            t_stims(n,1) = valid_stims(b);
            p_detect(n,1) = ground_truth(b);
        end
        % Remove overlapping
        [~,u_idx] = unique(t_stims);
        t_stims = t_stims(u_idx);
        p_detect = p_detect(u_idx);

        [pred_t, ~] = ConstantSimulation(psych, t_stims, p_detect, n_repeats(r), n_perms, 0.5,...
                                                n_sig_attempts, true, t, j, t);
        constant_t_j_bounded_reps(r,p,:) = pred_t;
        send(pfwb_update, 0);
    end
end

%% __ Plotting
cr = [.15 1]; 
sfig2 = figure('Name', 'Stimulus coverage');

err = rmse(constant_t_j,t,3) / s;
imAlpha=ones(size(err));
imAlpha(isnan(err)) = 0;
n_idx = all(isnan(err),1);
err(:,~n_idx) = imgaussfilt(err(:,~n_idx),1);

sf1_sp1 = subplot(2,3,1, 'Parent', sfig2);
    imagesc(point_density, point_range, err, 'Parent', sf1_sp1,'AlphaData',imAlpha)
    xlabel('Sigmoid Coverage'); ylabel('# Stimulus Levels')
    set(gca, 'ColorScale', 'log')
    caxis(cr); xticks([.1:.4:.9])
    
sf1_sp2 = subplot(2,3,2, 'Parent', sfig2);
    imagesc(point_density, point_range, imgaussfilt(rmse(constant_t_j_bounded,t,3)/s,1), 'Parent', sf1_sp2)
    caxis(cr);
    set(sf1_sp2, 'ColorScale', 'log')
    xlabel('Sigmoid Coverage');
    yticklabels({}); xticks([.1:.4:.9])

sf1_sp3 = subplot(2,3,3, 'Parent', sfig2);
    imagesc(n_repeats, point_range, imgaussfilt(rmse(constant_t_j_bounded_reps,t,3)/s,1)', 'Parent', sf1_sp3)
    caxis(cr);
    set(sf1_sp3, 'ColorScale', 'log', 'XLim', [0.5 20])
    xlabel('# Reps @ Intensity');
    yticklabels({})

cbar_axes = axes('Position', [.925 .6 .025 .325], 'Parent', sfig2);
    imagesc(1, linspace(cr(1), cr(2), 255), linspace(cr(1), cr(2), 255)')
    set(cbar_axes, 'YDir', 'normal', 'YAxisLocation', 'right', 'YTickLabelRotation', -90)
    xticks([]);yticks([])
    text(1.75,cr(2), num2str(cr(2)), 'Parent', cbar_axes, 'FontSize', 12)
    text(1.75,cr(1), num2str(cr(1)), 'Parent', cbar_axes, 'FontSize', 12)
    text(1.75,mean(cr), 'Error (Z)', 'Rotation', -90, 'Parent', cbar_axes, 'FontSize', 12, ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom')
    
sf1_sp4 = subplot(2,3,[4,5]); hold on
    pr = repmat(point_range, [length(n_repeats),1]);
    np = repmat(n_repeats, [length(point_range),1]);
    nt = np'.*pr;
    er = rmse(constant_t_j_bounded_reps,t,3)/s;
    scatter(nt, er, 'Parent', sf1_sp4, 'MarkerEdgeColor', 'none', 'MarkerFaceColor', [.6 .6 .6],...
        'MarkerFaceAlpha', 0.2)
    set(sf1_sp4, 'XLim',[0 200], 'YScale', 'log', 'YTick', [.15 .3 .6 1.2])
    xlabel('# Trials', 'Parent', sf1_sp4); ylabel('Error (Z)', 'Parent', sf1_sp4)

set(sfig2, 'Units', 'pixels', 'Position', OSScreenSize([9.32, 6], 'in', 1));
%ffname = fullfile(export_folder, 'DT_JNDOptimization');
%print(gcf,ffname, '-dpng', '-r600')


%% 2. Constrained Method of Constant Stimulus
t = 50; j = 5; s = jnd2sigma(j);

n_perms = 1e3; % (mr = 20) 1e2 = 12s, 1e3 = 2 mins, 1e4 = 25 mins, 1e5 = long
max_repeats = 40; % Number of stimuli at each intensity
num_points = 6;
num_afc = 2;
n_sig_attempts = 5;
target_method = 'Constrained'; % Constrained/Unconstrained
bound_logit = 1;

% Create CDF/sigmoid
ground_truth = getpdetect(valid_stims, t, s);
% Get appropriate stims
[target_stims, p_detect] = GetConstantTargets(ground_truth, valid_stims, target_method, num_points);

[wb, pfwb_update]  = ParforWaitbar('Constant simulation', max_repeats);
[t_repeats_constrained, jnd_repeats] = deal(zeros([n_perms, max_repeats]));
parfor r = 1:max_repeats
    [t_repeats_constrained(:,r), jnd_repeats(:,r)] = ConstantSimulation(psych, target_stims, p_detect, r, n_perms,...
        1/num_afc, n_sig_attempts, bound_logit, t, j, t);
    send(pfwb_update, 0);
end

% Constrained Staircase (start at correct t)
[amplitude_history, detection_idx, reversion_idx] = StaircaseSimulation(t, s, valid_stims, 2,...
        [3,1], 'Convergent', max_repeats*num_points*2, n_perms, t, 4, true);
[pred_t_constrained, trial_count_constrained, pred_t_sorted_constrained, trial_count_x_constrained] =...
    EstimateStaircaseThreshold(amplitude_history, reversion_idx,...
'Reversion', 5);

%% __ Plotting
fig3 = figure('Name', 'Constrained vs Staircase');
f3_sp1 = subplot(1,3,1, 'Parent', fig3); hold on
    plot([num_points,max_repeats *num_points], [t, t], 'Color', [0.6 0.6 0.6],...
            'LineStyle', '--', 'Parent', f3_sp1)
    AlphaLine([num_points:num_points:max_repeats*num_points], t_repeats_constrained', constant_col1,...
        'ErrorType', 'STD', 'Parent', f3_sp1, 'EdgeAlpha', 0)
    ylabel('Threshold Estimate'); 
    xticks([12,60,240]); xlabel('# Trials'); 
    set(gca, 'XScale', 'log')
    ylim([0 100]); xlim([10 max_repeats *num_points]); yticks([0:25:100])
    
f3_sp2 = subplot(1,3,2, 'Parent', fig3); hold on
    target_t = norminv(GetStaircaseTarget(2, [3,1]), t, s);
    plot([trial_count_x(2),max_repeats *num_points], [target_t, target_t], 'Color', [0.6 0.6 0.6],...
            'LineStyle', '--', 'Parent', f3_sp2)
    AlphaLine(trial_count_x, pred_t_sorted, staircase_col, 'ErrorType', 'STD', 'Parent', f3_sp2, 'EdgeAlpha', 0)
    ylim([0 100]); xlim([0 max_repeats *num_points]); yticks([0:25:100]); yticklabels({})
    set(gca, 'XScale', 'log');xticks([12,60,240]); xlabel('# Trials'); 

f3_sp3 = subplot(1,3,3, 'Parent', fig3); hold on
    % Constrained constant
    plot([1:max_repeats]*num_points, rmse(t_repeats_constrained,t,1)/s,...
        'Color', constant_col1, 'Parent', f3_sp3, 'LineWidth', 2)
    % Unconstrained constant
    plot([1:20]*12, rmse(t_repeats,t,1)/s,...
        'Color', constant_col1, 'Parent', f3_sp3, 'LineWidth', 2, 'LineStyle', '--')
    % Constrained constant
    plot(trial_count_x_constrained, rmse(pred_t_sorted_constrained, target_t, 1)/s,...
        'Color', staircase_col, 'Parent', f3_sp3, 'LineWidth', 2)
    % Unconstrained constant
    plot(trial_count_x, rmse(pred_t_sorted, target_t, 1)/s,...
        'Color', staircase_col, 'Parent', f3_sp3, 'LineWidth', 2, 'LineStyle', '--')
    
    set(f3_sp3, 'XScale', 'log', 'YScale', 'log', 'XLim', [10 max_repeats *num_points])
    xlabel('# Trials'); xticks([12,60,240]);
    ylabel('Error (Z)'); yticks([.1 .25 .5 1 2 4]); ylim([.1 6.5])

set(fig3, 'Units', 'pixels', 'Position', OSScreenSize([9.32, 3.74], 'in', 1));
ffname = fullfile(export_folder, 'DT_ConstrainedComparison');
%print(fig3, ffname, '-dpng', '-r600')


%% 3. Error Across JND/Threshold
n_perms = 1e2; % (mr = 20) 1e2 = 12s, 1e3 = 2 mins, 1e4 = 25 mins, 1e5 = long
num_repeats = 10; % Number of stimuli at each intensity
num_points = 5;
num_afc = 2;
n_sig_attempts = 5;
target_method = 'Constrained'; % Constrained/Unconstrained
bound_logit = 1;

t_range = [2:2:98];
j_range = [5:1:50];

st_p = GetStaircaseTarget(2, [3,1]);

[constant_tj, stair_tj] = deal(NaN([length(t_range), length(j_range), n_perms]));
constant_target = repmat(repmat(t_range', [1, length(j_range)]), [1,1,n_perms]);
staircase_target = NaN(size(constant_target));

[wb, pfwb_update]  = ParforWaitbar('3. Running simulation', length(t_range)*length(j_range));
for t = 1:length(t_range)
    temp_threshold = t_range(t);
    parfor j = 1:length(j_range)
        %global jng2sigma getpdetect
        temp_jnd = j_range(j);
        temp_sigma = jnd2sigma(temp_jnd);
        % Create CDF/sigmoid
        ground_truth = getpdetect(valid_stims, temp_threshold, temp_sigma);
        % Get appropriate stims
        [target_stims, p_detect] = GetConstantTargets(ground_truth, valid_stims, target_method, num_points);
        % Run constrained constant simulation
        [temp_constant_pred_t, ~] = ConstantSimulation(psych, target_stims, p_detect, num_repeats, n_perms,...
                1/num_afc, n_sig_attempts, bound_logit, temp_threshold, temp_jnd, temp_threshold);
        constant_tj(t,j,:) = temp_constant_pred_t;
        % Run constrained staircase simulation
        [temp_amplitude_history, ~, temp_reversion_idx] = StaircaseSimulation(...
            temp_threshold, temp_sigma, valid_stims, 2, [3,1], 'Convergent', num_repeats*num_points,...
            n_perms, temp_threshold, 4, true);
        [~, ~, temp_stair_pred_t, temp_stair_trialx] =...
            EstimateStaircaseThreshold(temp_amplitude_history, temp_reversion_idx, 'Reversion', 5);
        stair_tj(t,j,:) = temp_stair_pred_t(1:n_perms,end);
        send(pfwb_update, 0);
        staircase_target(t,j,:) = norminv(st_p, temp_threshold, temp_sigma);
    end
end

%% Compute error at each point
constant_target_vec = constant_target(:,:,1);
staircase_target_vec = staircase_target(:,:,1);
constant_pred = mean(constant_tj,3,'omitnan'); 
stair_pred = mean(stair_tj,3,'omitnan');

st = staircase_target(:);
sp = stair_tj(:);
ct = constant_target(:);
cp = constant_tj(:);

stx = linspace(min(st, [], 'all'), max(st, [], 'all'), 100);
sty = zeros([n_perms, length(stx)]);
for i = 1:length(stx)-1
    idx = find(sp > stx(i) & sp < stx(i+1));
    if ~isempty(idx)
        sty(1:length(idx),i) = sp(idx) - st(idx);
    end
end
sty(sty == 0) = NaN;

ctx = linspace(min(ct, [], 'all'), max(ct, [], 'all'), 100);
cty = zeros([n_perms, length(ctx)]);
for i = 1:length(ctx)-1
    idx = find(cp > ctx(i) & cp < ctx(i+1));
    if ~isempty(idx)
        cty(1:length(idx),i) =  ct(idx) - cp(idx);
    end
end
cty(cty == 0) = NaN;

hist_min = min([min(sty), min(cty)]);
hist_max = max([max(sty), max(cty)]);
hist_lim = ceil(max(abs([hist_min, hist_max])));
hist_bins = linspace(-hist_lim, hist_lim, 100);
hist_x = hist_bins(1:end-1) + range(hist_bins(1:2))/2;

st_hist = histcounts(sty, hist_bins); st_hist = st_hist / sum(st_hist);
ct_hist = histcounts(cty, hist_bins); ct_hist = ct_hist / sum(ct_hist);

%% __ Plotting
figure;
subplot(1,3,1); hold on
    scatter(constant_target_vec(:), constant_pred(:), 30,...
        'MarkerFaceColor', constant_col1, 'MarkerEdgeColor', constant_col1, 'MarkerFaceAlpha', 0.1)
    scatter(staircase_target_vec(:), stair_pred(:), 30,...
        'MarkerFaceColor', staircase_col, 'MarkerEdgeColor', staircase_col, 'MarkerFaceAlpha', 0.1)
    plot([0 100], [0 100], 'Color', [.6 .6 .6], 'LineStyle', '--')
    
    xlabel('Actual Threshold'); ylabel('Estimated Threshold')
    
subplot(1,3,2); hold on
    plot([2, 98], [0 0], 'Color', [.6 .6 .6], 'LineStyle', '--')
    AlphaLine(stx, sty, staircase_col)
    AlphaLine(ctx, cty, constant_col1)
    
    xlabel('Threshold'); ylabel('Residual Error')

subplot(1,3,3); hold on
    stairs(hist_x, st_hist, 'Color', staircase_col)
    stairs(hist_x, ct_hist, 'Color', constant_col1)
    xlabel('Residual Error'); ylabel('Probability')
    xlim([-50 50])
    

%% PEST
threshold = 50; j = 5; s = jnd2sigma(j);
ground_truth = getpdetect(valid_stims, threshold, s);

num_afc = 2;
init_amp = valid_stims(randi(length(valid_stims))); stim_amp = init_amp;
max_step = 20;
init_step = 10; step_size = init_step;
max_trials = 50;
target_p = .70;
wald_stat = .0;
current_dir = -1; new_dir = -1; last_rev_was_double = false;
sequential_dir_count = 0; last_step_was_double = false;
W = 1; % Wald value
Lt = NaN([10,1]); lti = 1;
minLt = 4;

real_time_plot = false;
if real_time_plot
    f = figure;
    ax1 = subplot(1,2,1, 'Parent', f); hold on
    ax2 = subplot(1,2,2, 'Parent', f); hold on
    set(ax1, 'XLim', [0 max_trials])
    set(ax2, 'XLim', [0 max_trials])
end

opts = optimset('Display','off'); % Disable reporting for lsqcurvefit
noise_draws = rand([1,max_trials]);
[amplitude_history, detection_history, reversion_history, p_hist] = deal(NaN(size(noise_draws)));
for t = 1:max_trials
    amplitude_history(t) = stim_amp;
    [~, stim_idx] = min(abs(valid_stims - stim_amp));
    stim_draw = ground_truth(stim_idx);
    
    % Compare noise & stim draws + keep track of consecutive answers
    if stim_draw > noise_draws(t) || (num_afc > 1 && rand(1) < 1/num_afc)
        detection_history(t) = 1;
        Lt(lti) = 1;
        lti = lti + 1;
    else
        detection_history(t) = 0;
        Lt(lti) = 0;
        lti = lti + 1;
    end
    
    % Evaluate likelihood
    if sum(~isnan(Lt)) >= minLt % Only check if minimum number of trials has been reached
        NumDt = sum(Lt, 'omitnan'); % Observed detections
        expNumDt = sum(~isnan(Lt)) * target_p; % Expected detections
        bounds = [expNumDt - W, expNumDt + W]; % Wald-adjusted bounds
        if NumDt < bounds(1)
            new_dir = 1;
            stim_amp = stim_amp + step_size;
            % Reset current amplitude tracker
            Lt = NaN([10,1]); lti = 1;
        elseif NumDt > bounds(2)
            new_dir = -1;
            stim_amp = stim_amp - step_size;
            % Reset current amplitude tracker
            Lt = NaN([10,1]); lti = 1;
        end
    end
 
    % Enture stim is in range
    if stim_amp < min(valid_stims)
        stim_amp = min(valid_stims);
    elseif stim_amp > max(valid_stims)
        stim_amp = max(valid_stims);
    else
        [~, stim_idx] = min(abs(valid_stims - stim_amp));
        stim_amp = valid_stims(stim_idx);
    end
    
    % Get new update rule
    if isempty(Lt)
        if new_dir ~= current_dir
            current_dir = new_dir;
            sequential_dir_count = 0;
            if last_step_was_double
                last_rev_was_double = true;
            else
                last_rev_was_double = false;
            end
        else
            sequential_dir_count = sequential_dir_count + 1;
        end
        
        % Determine step size - PEST Rules
        if sequential_dir_count == 0
            step_size = step_size / 2; % Half step size on direction change
            last_step_was_double = false;
        elseif sequential_dir_count == 1
            %step_size = step_size; % Maintain on subsequent for same direction
            last_step_was_double = false;
        elseif sequential_dir_count == 2 && last_rev_was_double
            step_size = step_size * 2; % Only double if the last reversion was from a double
            last_step_was_double = true;
        elseif sequential_dir_count == 2 && ~last_rev_was_double
            %step_size = step_size; % Maintain on subsequent for same direction
            last_step_was_double = false;
        elseif sequential_dir_count > 2
            step_size = step_size * 2; % Double away
            last_step_was_double = true;
        end

        % Ensure step size is in range
        if step_size < min(valid_stims)
            step_size = min(valid_stims);
        elseif step_size > max_step
            step_size = max_step;
        end
    end
end


if ~real_time_plot
clf;
subplot(1,3,[1,2]); hold on
    x = 1:max_trials;
    scatter(x(logical(detection_history)), amplitude_history(logical(detection_history)), 30, [0 1 0], 'filled')
    scatter(x(~logical(detection_history)), amplitude_history(~logical(detection_history)), 30, [1 0 0], 'filled')
    ylim([0 100])
    
subplot(1,3,3); hold on
    ua = unique(amplitude_history);
    pd = zeros(size(ua));
    for i = 1:length(ua)
        idx = amplitude_history == ua(i);
        pd(i) = sum(detection_history(idx)) / sum(idx);
    end
    pd = (pd - .5) * 2;
    [sig_c, ~] = lsqcurvefit(sigfun,[sigma2k(s),threshold],[0,ua,2*max(ua)], [0,pd,1], [],[],opts);
    plot(valid_stims, ground_truth)
    plot(valid_stims, sigfun(sig_c, valid_stims))
end
%% Functions
return
function [target_stims, p_detect] = GetConstantTargets(ground_truth, valid_stims, target_method, num_points)
    % Get uniformly distributed (in probability space) stim levels
    if strcmpi(target_method, 'Constrained')
        target_p = linspace(0.2,.8, num_points)';
        [target_stims, p_detect] = deal(zeros(size(target_p)));
        for n = 1:num_points
            [a,b] = min(abs(ground_truth-target_p(n)));
            target_stims(n,1) = valid_stims(b);
            p_detect(n,1) = ground_truth(b);
        end
        
    elseif strcmpi(target_method, 'Unconstrained')
        t_idx = round(linspace(1, length(valid_stims), num_points));
        target_stims = valid_stims(t_idx)';
        p_detect = ground_truth(t_idx)';
    end

    % Check if stim range has a floor or ceiling or has duplicates
    if length(unique(target_stims)) ~= length(target_stims)
       [~,u_idx] = unique(target_stims);
       target_stims = target_stims(u_idx);
       p_detect = p_detect(u_idx);
    end
end

function [pred_t, pred_jnd] = ConstantSimulation(psych, target_stims, p_detect, num_repeats, n_perms, chance,...
    n_sig_attempts, bound_logit, t, j, s0)
        
    opts = optimset('Display','off'); % Disable reporting for lsqcurvefit
    if length(target_stims) == 1 && ~bound_logit
        [pred_jnd, pred_t] = deal(NaN([n_perms, 1]));
        return
    else
        [pred_jnd, pred_t] = deal(zeros([n_perms, 1]));
    end
    [pred_jnd, pred_t] = deal(zeros([n_perms, 1]));
    for p = 1:n_perms
        % Get draws stimulus and random
        stim_draw = rand([length(target_stims), num_repeats]);
        afc_draw = rand(size(stim_draw));
        % Deterimine detection 
        detections = p_detect > stim_draw | afc_draw > 1 - chance;
        detections = mean(detections,2);
        detections = (detections-chance) * (1/chance);
        detections(detections<0) = 0;
        
        lb = [0,1];
        ub = [1 99];
        
        if s0 == 0
            s0 = randi([10,90]);
            k0 = rand(1);
        elseif s0 == t
            s0 = t + (randn(1)*j/5);
            k0 = psych.sigma2k(psych.jnd2sigma(j)) + (randn(1)*j/10);
        end
        
        % Fit a sigmoid with bounds
        sig_cs = cell([n_sig_attempts,1]);
        sig_fits = zeros([n_sig_attempts,1]);
        for s = 1:n_sig_attempts
            if bound_logit
                [sig_c, ~] = lsqcurvefit(psych.sigfun,[rand(1),randi([1,90])],...
                    [0;target_stims;2*max(target_stims)], [0;detections;1], lb, ub, opts);
            else
                [sig_c, ~] = lsqcurvefit(psych.sigfun,[k0,s0], target_stims, detections, lb, ub, opts);
            end
            sig_y = psych.sigfun(sig_c, target_stims);
            sig_cs{s} = sig_c;
            sig_fits(s) = mean(abs(sig_y - detections));
        end
        [~,sig_idx] = min(sig_fits);
        pred_t(p,1) = sig_cs{sig_idx}(2);
        
        % Determine estimated JND
        pred_jnd(p,1) = (psych.invsig(sig_cs{sig_idx},.75) - psych.invsig(sig_cs{sig_idx},.25))/2;
    end
end

function [amplitude_history, detection_idx, reversion_idx] = StaircaseSimulation(mu, JND, stims, num_afc,...
    consecutive_criterion, method, num_trials, n_perms, init_amp, max_step_size, force_consecutive)    
    % Convergent (1DB power ratio = 1.259; amplitude ratio = 1.122)
    pos_step_coeff = .75;
    neg_step_coeff = 1;
    % Fixed
    pos_step = 4;
    neg_step = 2;
    % Limits
    min_step_size = 2;
    %max_step_size = 20;
    jnd2sigma = @(j) (1/norminv(.75))*j;
    s = jnd2sigma(JND);

    % Task design
    chance = 1 / num_afc;
    
    [detection_idx, amplitude_history, reversion_idx] = deal(NaN([n_perms,num_trials]));
    parfor p = 1:n_perms  
        ground_truth_afc = normcdf(stims, mu, s); % Create CDF/sigmoid
        par_consecutive_criterion = consecutive_criterion; % Better if in parallel loop

        step_size = max_step_size;
        new_direction = 0;
        noise_draws = rand([1,num_trials]);
        if init_amp == 0
            stim_amp_idx = randi(length(stims),1);
            stim_amp = stims(stim_amp_idx);
        elseif init_amp == mu
            stim_amp = mu + (randn(1)*JND/5);
        else
            stim_amp = init_amp;
        end
        
        current_direction = 0;
        consecutive_answers = [0,0];
        for t = 1:num_trials
            % Check if stim is valid
            [~, stim_idx] = min(abs(stims - stim_amp));
            stim_draw = ground_truth_afc(stim_idx);
            amplitude_history(p,t) = stims(stim_idx);

            % Compare noise & stim draws + keep track of consecutive answers
            if stim_draw > noise_draws(t) || (num_afc > 1 && rand(1) < chance)
                detection_idx(p,t) = 1;
                consecutive_answers(1) = consecutive_answers(1) + 1;
                if force_consecutive
                    consecutive_answers(2) = 0;
                end
            else
                detection_idx(p,t) = 0;
                consecutive_answers(2) = consecutive_answers(2) + 1;
                if force_consecutive
                    consecutive_answers(1) = 0;
                end
            end

            % Only alter stimulus after criteria has been reached
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
                    if strcmpi(method, 'fixed')
                        step_size = neg_step;
                    elseif strcmpi(method, 'convergent')
                        step_size = step_size * neg_step_coeff;
                    end
                elseif new_direction == 1
                    if strcmpi(method, 'fixed')
                        step_size = pos_step;
                    elseif strcmpi(method, 'convergent')
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
                reversion_idx(p,t) = new_direction;
            end

            % Check amplitude is in stim range
            % Also add trial as a reversion idx if floor/ceiling
            if stim_amp < min(stims)
               stim_amp = min(stims);
               reversion_idx(p,t) = current_direction;
            elseif stim_amp > max(stims)
               stim_amp = max(stims);
               reversion_idx(p,t) = current_direction;
            end
        end
    end
end

function [pred_t, trial_count, pred_t_sorted, trial_count_x] = EstimateStaircaseThreshold(amplitude_history, reversion_idx,...
    trial_or_reversion, stop_criteria)
    [num_perms, num_trials] = size(amplitude_history);
    if strcmpi(trial_or_reversion, 'Reversion')
        max_reversions = max(sum(~isnan(reversion_idx),2,'omitnan'));
        [pred_t, trial_count] = deal(nan([num_perms, max_reversions-stop_criteria+1]));
        for p = 1:num_perms
            r_idx = ~isnan(reversion_idx(p,:));
            reversions_in_perm = sum(r_idx);
            rp_crit = reversions_in_perm - stop_criteria + 1;
            for c = 1:rp_crit
                rc_idx = find(r_idx, c+stop_criteria-1, 'first');
                rc_amps = amplitude_history(p,rc_idx);
                pred_t(p,c) = mean(rc_amps);
                trial_count(p,c) = rc_idx(end);
            end
        end
        % Prep output for AlphaLine
        trial_count_bins = [0:5:max(trial_count, [], 'all', 'omitnan')];
        trial_count_x = trial_count_bins(2:end);
        bin_counts = histcounts(trial_count, trial_count_bins);
        pred_t_sorted = NaN([max(bin_counts), length(trial_count_bins)-1]);
        for i = 1:length(trial_count_bins)-1
            ii =  trial_count > trial_count_bins(i) & trial_count < trial_count_bins(i+1);
            pred_t_sorted(1:sum(ii(:)), i) = pred_t(ii);
        end
        
    elseif strcmpi(trial_or_reversion, 'Trial')
        [pred_t, trial_count] = deal(nan([num_perms, num_trials - stop_criteria]));
        for p = 1:num_perms
            for t = 1:num_trials - stop_criteria
                pred_t(p,t) = mean(amplitude_history(p,t:t+stop_criteria));
                trial_count(p,t) = t+stop_criteria-1;
            end
        end
        
        pred_t_sorted = pred_t;
        trial_count_x = trial_count(1,:);
    else
        return
    end
end

function target_lvl = GetStaircaseTarget(num_afc, consecutive_criterion)
    chance = 1/num_afc;
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
end
