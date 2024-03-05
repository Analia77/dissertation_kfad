% Be sure to change DATA_DIR to write to a directory if you so desire.
DATA_DIR = "/Users/analiacabello/Desktop/Y5/DISS/2D_matlab_experiments";

% Include utilities dir.
addpath(fullfile(pwd, "../util"));

% Include integrator dir
addpath(fullfile(pwd, "../integrators/scheqleapfrog/matlab"));


% Filesystem setup
PLOT = 0;
SAVE_PLOT_TO_FS = 1;
SAVE_DATA_TO_FS = 1;
%% 2D Setup
dim = 2;
num_cells = 32;

% Run the experiment setup script to init the experiment objects
experimentSetup;

%%

experiments = [...
    ackley_experiment, ...
    ackley2_experiment, ...
    alpine1_experiment, ...
    alpine2_experiment ...
    bohachevsky2_experiment, ...
    camel3_experiment, ...
    csendes_experiment, ...
    defl_corr_spring_experiment, ...
    dropwave_experiment, ...
    easom_experiment, ...
    griewank_experiment, ...
    holder_experiment, ...
    hosaki_experiment, ...
    levy_experiment, ...
    levy13_experiment, ...
    michalewicz_experiment, ...
    rastrigin_experiment, ...
    rosenbrock_experiment, ...
    shubert_experiment, ...
    styblinski_tang_experiment, ...
    sumofsquares_experiment, ...
    xinsheyang3_experiment, ...
];

%% Experiment Loops

syms x1 x2;


parfor tid = 1:numel(experiments)
    experiment = experiments(tid);
    experiment_dir = experiment.experiment_dir;
    experiment_V_str = experiment.experiment_V_str;
    
    fprintf("thread %d, working on experiment %s\n", tid, experiment_V_str);
    
    % Prep directory
    if SAVE_PLOT_TO_FS || SAVE_DATA_TO_FS
        % Comment at the end of the following line suppresses a warning
        % about potentially unreachable code (intentional switch on
        % WRITE_TO_FS).
        save_target_dir = fullfile(DATA_DIR, experiment_dir); %#ok<*UNRCH>

        if ~exist(save_target_dir, "dir")
            mkdir(save_target_dir);
        end
        % Silence figures if writing
        if PLOT
            fig = figure(tid); set(fig,'visible','off');
        end
    else
        fig = figure(tid); set(fig,'visible','on');
    end
    
    % Classical setup
    N = 1000;
    MAX_FRAMES = 1000;
    cap_frame_every = 10;
    eta = 5e-2; % stepsize
    rng('default');
    starting_points = rand(2, N);

    % RUN SGD (Noisy GD)
    fprintf("thread %d, working on exp %s, starting classical SGD \n", tid, experiment_V_str);
    ngd_fn_vals = zeros(N, MAX_FRAMES);
    ngd_positions = zeros(N, MAX_FRAMES, 2);
    ngd_last_frame = zeros(N, 1);

    % Distribution of noise
    mu = 0;
    sigma = 1;

    for idx = 1:N
        if mod(idx-1, 100) == 0
            disp(idx-1);
        end

        x = starting_points(:, idx);
        grad = experiment.eval_grad(x(1), x(2));

        % Convergence parameter in gradient, epsilon
        conv_eps = 1e-8;

        step_count = 0;
        frame_count = 0;

        %while norm(grad) > conv_eps && frame_count < MAX_FRAMES
        while frame_count < MAX_FRAMES

            x = x - eta * (grad + normrnd(mu, sigma, 2, 1));

            % Project back inside [0,1]x[0,1] boundaries
            if x(1) < 0
                x(1) = 0;
            end

            if x(2) < 0
                x(2) = 0;
            end

            if x(1) > 1
                x(1) = 1;
            end

            if x(2) > 1
                x(2) = 1;
            end

            grad = experiment.eval_grad(x(1), x(2));

            step_count = step_count + 1;

            if mod(step_count, cap_frame_every) == 0
                frame_count = frame_count + 1;
                res_idx = step_count / cap_frame_every;
                ngd_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                ngd_positions(idx, res_idx, :) = x;
                ngd_last_frame(idx) = frame_count;
            end
        end

        if frame_count == 0
            ngd_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
            ngd_positions(idx, 1, :) = x;
            ngd_last_frame(idx) = 1;
        end
    end

    ngd_fn_vals = ngd_fn_vals(:, 1:max(ngd_last_frame));
    ngd_positions = ngd_positions(:, 1:max(ngd_last_frame), :);

    if SAVE_DATA_TO_FS
        parsave_sgd(experiment_dir, save_target_dir, ngd_fn_vals, ngd_positions, ngd_last_frame);
    end

    fprintf("thread %d, working on exp %s, finished classical SGD\n", tid, experiment_V_str);


    % % Run NESTEROV
    % % http://www.princeton.edu/~yc5/ele522_optimization/lectures/accelerated_gradient.pdffprintf("thread %d, working on exp %s, starting classical Polyak\n", tid, experiment_V_str);
    % fprintf("thread %d, working on exp %s, starting classical Nesterov\n", tid, experiment_V_str);
    % 
    % nesterov_fn_vals = zeros(N, MAX_FRAMES);
    % nesterov_positions = zeros(N, MAX_FRAMES, 2);
    % nesterov_last_frame = zeros(N, 1);
    % 
    % for idx = 1:N
    %     if mod(idx-1, 100) == 0
    %         disp(idx-1);
    %     end
    % 
    %     % initial values (selected randomly earlier)
    %     x = starting_points(:, idx);
    %     y = starting_points(:, idx);
    %     x_last = starting_points(:, idx);
    %     y_last = starting_points(:, idx);
    % 
    %     grad_y = experiment.eval_grad(y(1), y(2));
    % 
    %     % Convergence in gradient, epsilon
    %     conv_eps = 1e-8;
    % 
    %     step_count = 0;
    %     frame_count = 0;
    % 
    %     %while norm(grad_y) > conv_eps && frame_count < 10000
    %     while frame_count < MAX_FRAMES
    %         x_last = x;
    %         y_last = y;
    % 
    %         x = y - eta * grad_y;
    % 
    %         % Project back inside [0,1]x[0,1] boundaries
    %         if x(1) < 0
    %             x(1) = 0;
    %         end
    % 
    %         if x(2) < 0
    %             x(2) = 0;
    %         end
    % 
    %         if x(1) > 1
    %             x(1) = 1;
    %         end
    % 
    %         if x(2) > 1
    %             x(2) = 1;
    %         end
    % 
    %         y = x + (step_count / (step_count + 3)) * (x - x_last);
    % 
    %         % Project back inside [0,1]x[0,1] boundaries
    %         if y(1) < 0
    %             y(1) = 0;
    %         end
    % 
    %         if y(2) < 0
    %             y(2) = 0;
    %         end
    % 
    %         if y(1) > 1
    %             y(1) = 1;
    %         end
    % 
    %         if y(2) > 1
    %             y(2) = 1;
    %         end
    % 
    %         grad_y = experiment.eval_grad(y(1), y(2));
    % 
    %         step_count = step_count + 1;
    % 
    %         if mod(step_count, cap_frame_every) == 0
    %             frame_count = frame_count + 1;
    %             res_idx = step_count / cap_frame_every;
    %             nesterov_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
    %             nesterov_positions(idx, res_idx, :) = y;
    %             nesterov_last_frame(idx) = frame_count;
    %         end
    %     end
    % 
    %     if frame_count == 0
    %         nesterov_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
    %         nesterov_positions(idx, 1, :) = x;
    %         nesterov_last_frame(idx) = 1;
    %     end
    % end
    % 
    % nesterov_fn_vals = nesterov_fn_vals(:, 1:max(nesterov_last_frame));
    % nesterov_positions = nesterov_positions(:, 1:max(nesterov_last_frame), :);
    % 
    % if SAVE_DATA_TO_FS
    %     parsave_nesterov(experiment_dir, save_target_dir, nesterov_fn_vals, nesterov_positions, nesterov_last_frame);
    % end
    % 
    % fprintf("thread %d, working on exp %s, finished classical Nesterov\n", tid, experiment_V_str);
    % 

    % Run the LDHD Optimization Method
    ldhd_fn_vals = zeros(N, MAX_FRAMES);
    ldhd_positions = zeros(N, MAX_FRAMES, 2);
    ldhd_last_frame = zeros(N, 1);

    % Set up necessary parameters
    gamma = 0.2;
    
    for idx = 1:N
        % Initialization and other setup for the new method
        % ...
        % ask about this
        if mod(idx-1, 100) == 0
            disp(idx-1);
        end
        x = starting_points(:, idx);
        % need to initialize p
        p = rand(size(x));
        grad = experiment.eval_grad(x(1), x(2));

        % Convergence parameter in gradient, epsilon
        conv_eps = 1e-8;

        step_count = 0;
        frame_count = 0;

        %while norm(grad) > conv_eps && frame_count < MAX_FRAMES
        while frame_count < MAX_FRAMES
            % execute the LDHD method using the symmetric integrator BADAB
            [x, p] = ldhd(x, p, eta, -(grad+ normrnd(mu, sigma, 2, 1)), gamma);

            % Project back inside [0,1]x[0,1] boundaries
            if x(1) < 0
                x(1) = 0;
            end

            if x(2) < 0
                x(2) = 0;
            end

            if x(1) > 1
                x(1) = 1;
            end

            if x(2) > 1
                x(2) = 1;
            end

            grad = experiment.eval_grad(x(1), x(2));

            step_count = step_count + 1;

            if mod(step_count, cap_frame_every) == 0
                frame_count = frame_count + 1;
                res_idx = step_count / cap_frame_every;
                ldhd_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                ldhd_positions(idx, res_idx, :) = x;
                ldhd_last_frame(idx) = frame_count;
            end
        end

        if frame_count == 0
            ldhd_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
            ldhd_positions(idx, 1, :) = x;
            ldhd_last_frame(idx) = 1;
        end
    end

    ldhd_fn_vals = ldhd_fn_vals(:, 1:max(ldhd_last_frame));
    ldhd_positions = ldhd_positions(:, 1:max(ldhd_last_frame), :);

    
    % Save Data for the New Method
    if SAVE_DATA_TO_FS
    parsave_ldhd(experiment_dir, save_target_dir, ldhd_fn_vals, ldhd_positions, ldhd_last_frame);
    end
    
    fprintf("thread %d, working on exp %s, finished the LDHD method\n", tid, experiment_V_str);

    % Run the KFAD Optimization Method
    kfad_fn_vals = zeros(N, MAX_FRAMES);
    kfad_positions = zeros(N, MAX_FRAMES, 2);
    kfad_last_frame = zeros(N, 1);

    % Set up necessary parameters
    % gamma -> this is already set up for ldhd and it is identical to the
    % ones we use for kfad
    alpha = 1;
    mu_kfad = 1;
    
    
    for idx = 1:N
        % Initialization and other setup for the new method
        % ...
        if mod(idx-1, 100) == 0
            disp(idx-1);
        end
        x = starting_points(:, idx);
        % need to initialize p
        p = rand(size(x));
        % need to initialize xi
        xi = 0;
        grad = experiment.eval_grad(x(1), x(2));

        % Convergence parameter in gradient, epsilon
        conv_eps = 1e-8;

        step_count = 0;
        frame_count = 0;
    
        %while norm(grad) > conv_eps && frame_count < MAX_FRAMES
        while frame_count < MAX_FRAMES
            % execute the KFAD method using the symmetric integrator BADAB
            [x, p, xi] = kfad(x, p, xi, eta, -(grad+ normrnd(mu, sigma, 2, 1)), gamma, alpha, mu_kfad);

            % Project back inside [0,1]x[0,1] boundaries
            if x(1) < 0
                x(1) = 0;
            end

            if x(2) < 0
                x(2) = 0;
            end

            if x(1) > 1
                x(1) = 1;
            end

            if x(2) > 1
                x(2) = 1;
            end

            grad = experiment.eval_grad(x(1), x(2));

            step_count = step_count + 1;

            if mod(step_count, cap_frame_every) == 0
                frame_count = frame_count + 1;
                res_idx = step_count / cap_frame_every;
                kfad_fn_vals(idx, res_idx) = experiment.eval_fn(x(1), x(2));
                kfad_positions(idx, res_idx, :) = x;
                kfad_last_frame(idx) = frame_count;
            end
        end

        if frame_count == 0
            kfad_fn_vals(idx, 1) = experiment.eval_fn(x(1), x(2));
            kfad_positions(idx, 1, :) = x;
            kfad_last_frame(idx) = 1;
        end
    end

    kfad_fn_vals = kfad_fn_vals(:, 1:max(kfad_last_frame));
    kfad_positions = kfad_positions(:, 1:max(kfad_last_frame), :);
    
    % Save Data for the New Method
    if SAVE_DATA_TO_FS
        parsave_kfad(experiment_dir, save_target_dir, kfad_fn_vals, kfad_positions, kfad_last_frame);
    end
    
    fprintf("thread %d, working on exp %s, finished the KFAD method\n", tid, experiment_V_str);
end % N samples



function parsave_sgd(experiment_dir, save_target_dir, ngd_fn_vals, ngd_positions, ngd_last_frame)
    save_target_fname = strcat(experiment_dir, "_SGD");
    save_target_path = fullfile(save_target_dir, save_target_fname);
    save(save_target_path, 'ngd_fn_vals', 'ngd_positions', 'ngd_last_frame');
end

function parsave_nesterov(experiment_dir, save_target_dir, nesterov_fn_vals, nesterov_positions, nesterov_last_frame)
    save_target_fname = strcat(experiment_dir, "_NAGD");
    save_target_path = fullfile(save_target_dir, save_target_fname);
    save(save_target_path, 'nesterov_fn_vals', 'nesterov_positions', 'nesterov_last_frame');
end

function parsave_ldhd(experiment_dir, save_target_dir, ldhd_fn_vals, ldhd_positions, ldhd_last_frame)
    save_target_fname = strcat(experiment_dir, "_LDHD");
    save_target_path = fullfile(save_target_dir, save_target_fname);
    save(save_target_path, 'ldhd_fn_vals', 'ldhd_positions', 'ldhd_last_frame');
end

function parsave_kfad(experiment_dir, save_target_dir, kfad_fn_vals, kfad_positions, kfad_last_frame)
    save_target_fname = strcat(experiment_dir, "_KFAD");
    save_target_path = fullfile(save_target_dir, save_target_fname);
    save(save_target_path, 'kfad_fn_vals', 'kfad_positions', 'kfad_last_frame');
end

function [x1, p1] = splitA_LDHD(x0, p0, timestep)
    % Splitting part A
    x1 = x0 + timestep * p0;
    p1 = p0;
end

function [x1, p1] = splitB_LDHD(x0, p0, F, timestep)
    % Splitting part B
    x1 = x0;
    p1 = p0 + timestep * F; % do i need to index here?
end

function [x1, p1] = splitD_LDHD(x0, p0, timestep, gamma)
    % Splitting part D
    x1 = x0;
    p1 = exp(-timestep * gamma) * p0;
end

function [x1, p1] = ldhd(x0, p0, timestep, F, gamma)
    % Symmetric integrator for LDHD (BADAB_step)
    
    % B - half a timestep
    [x0, p0] = splitB_LDHD(x0, p0, F, timestep / 2);
    
    % A - half a timestep
    [x0, p0] = splitA_LDHD(x0, p0, timestep / 2);
    
    % D
    [x0, p0] = splitD_LDHD(x0, p0, timestep, gamma);
    
    % A
    [x0, p0] = splitA_LDHD(x0, p0, timestep / 2);
    
    % B
    [x1, p1] = splitB_LDHD(x0, p0, F, timestep / 2);
end

function [x1, p1, e1] = splitA_KFAD(x0, p0, e0, timestep)
    % Splitting part A
    x1 = x0 + timestep * p0;
    p1 = p0;
    e1 = e0;
end

function [x1, p1, e1] = splitB_KFAD(x0, p0, e0, timestep, F)
    % Splitting part B
    x1 = x0;
    p1 = p0 + timestep * F;
    e1 = e0;
end

function [x1, p1, e1] = splitD_KFAD(x0, p0, e0, timestep, gamma)
    % Splitting part D
    x1 = x0;
    p1 = exp(-timestep * gamma) * p0;
    e1 = e0;
end

function [x1, p1, e1] = splitC_KFAD(x0, p0, e0, timestep, alpha, mu)
    % Splitting part C
    x1 = x0;
    p1 = exp(-(timestep/2) * e0) * p0;
    e1 = exp(-timestep * alpha) * e0 + (1/(mu * alpha)) * (1 - exp(-timestep * alpha)) * (p1' * p1);
    p1 = exp(-(timestep/2) * e1) * p1;
end

function [x1, p1, e1] = kfad(x, p, e, timestep, F, gamma, alpha, mu)
    % Symmetric integrator for KFAD
    
    % D
    [x, p, e] = splitD_KFAD(x, p, e, timestep/2, gamma);
    
    % A
    [x, p, e] = splitA_KFAD(x, p, e, timestep/2);
    
    % B
    [x, p, e] = splitB_KFAD(x, p, e, timestep/2, F);
    
    % C - only step evaluated at a full timestep instead of just half
    [x, p, e] = splitC_KFAD(x, p, e, timestep, alpha, mu);
    
    % B
    [x, p, e] = splitB_KFAD(x, p, e, timestep/2, F);
    
    % A
    [x1, p1, e1] = splitA_KFAD(x, p, e, timestep/2);
    
    % D
    [x1, p1, e1] = splitD_KFAD(x1, p1, e1, timestep/2, gamma);
end
