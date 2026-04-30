% --- CHƯƠNG TRÌNH MÔ PHỎNG SO SÁNH OUTAGE PROBABILITY (FAMA) ---
clc; clear; close all;
addpath('Core/')
% 1. CÀI ĐẶT THÔNG SỐ GỐC
W_val = 5;          % Chiều dài anten mục tiêu
N_val = 100;           % Số cổng mục tiêu
U = 3;                % Số người dùng
B = 12;                % Số khối (Số đầu ra của ML)
gamma_dB = -5:2:15;  % Dải SNR từ -5dB đến 10dB
num_iterations = 100;  % Số lần lặp mỗi điểm SNR để lấy trung bình

% Tải mạng Neural
if exist('trained_net_softmax.mat', 'file')
    load('trained_net_softmax.mat', 'net');
else
    error('Không tìm thấy file trained_net_softmax.mat. Hãy train mạng trước!');
end

% Khởi tạo mảng lưu kết quả
OP_ML_avg  = zeros(size(gamma_dB));
OP_BC_avg  = zeros(size(gamma_dB));
OP_Jakes = zeros(size(gamma_dB));

fprintf('Đang chạy mô phỏng (W=%.1f, N=%d, B=%d)...\n', W_val, N_val, B);

% 2. VÒNG LẶP MÔ PHỎNG THEO SNR
for k = 1:length(gamma_dB)
    SNR_lin = 10^(gamma_dB(k)/10);
    temp_ML = 0; temp_BC = 0; temp_Takes = 0;
    
    for iter = 1:num_iterations
        % Giả lập biến thiên nhẹ để dữ liệu khách quan
        W = W_val; 
        N = N_val;
        % --- CHỖ CẦN THAY THẾ ---
        d = W_val / (N_val - 1);           % Khoảng cách giữa 2 cổng kế tiếp
        mu_val = besselj(0, 2*pi*d); % Tính J0(2*pi*d)
        mu2 = mu_val^2;            % Bình phương để làm hệ số tương quan công suất
        % ------------------------
        
        % Tạo ma tran Jakes
        Sigma = toeplitz(besselj(0, 2*pi*(0:N-1)*W/(N-1)));
        lambda_true = sort(eig(Sigma), 'descend');

        % --- PHƯƠNG PHÁP 1: DEEP LEARNING ---
        input_ML = [W, U, SNR_lin, N];
        L_raw = predict(net, input_ML);
        
        % Hậu xử lý: Làm tròn và Bù trừ sai số dựa trên phần dư (Residual Balancing)
        L_ML = floor(L_raw);
        L_ML(L_ML < 1) = 1;
        diffN = N - sum(L_ML);
        if diffN > 0
            [~, sorted_idx] = sort(mod(L_raw, 1), 'descend');
            for j = 1:diffN
                target_idx = sorted_idx(mod(j-1, length(L_ML)) + 1);
                L_ML(target_idx) = L_ML(target_idx) + 1;
            end
        end
        temp_ML = temp_ML + CalcOutage(SNR_lin, L_ML, mu2, U, 'Quadrature', 20);

        % --- PHƯƠNG PHÁP 2: BLOCK CORRELATION (BASELINE) ---
        L_BC = BlockCorrelation(N, lambda_true, B, mu2);
        temp_BC = temp_BC + CalcOutage(SNR_lin, L_BC, mu2, U, 'Quadrature', 20);
    end
    


    % Lưu giá trị trung bình
    OP_ML_avg(k)  = temp_ML / num_iterations;
    OP_BC_avg(k)  = temp_BC / num_iterations;
    OP_Jakes(k) = SimOutage(1e6, SNR_lin, Sigma, U);
     fprintf('SNR = %2d dB | ML: %.2e | BC: %.2e | Jakes: %.2e\n', ...
            gamma_dB(k), OP_ML_avg(k), OP_BC_avg(k), OP_Jakes(k));
end


% 3. VẼ ĐỒ THỊ KẾT QUẢ
figure('Color', 'w', 'Position', [100, 100, 800, 600]);
semilogy(gamma_dB, OP_BC_avg, 'r--o', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'r'); hold on;
semilogy(gamma_dB, OP_ML_avg, 'b-s', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'b');
semilogy(gamma_dB, OP_Jakes, 'g-*', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'b');

grid on;
ax = gca;
ax.YMinorGrid = 'on';
xlabel('Average SNR \gamma (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Outage Probability (OP)', 'FontSize', 12, 'FontWeight', 'bold');
title(['Performance Comparison: FAMA Fluid Antenna (W=', num2str(W_val), ', N=', num2str(N_val), ')'], 'FontSize', 13);
legend('Block Correlation (Baseline)', 'Proposed Deep Learning', 'jakes (Brute-force)', 'Location', 'southwest');

fprintf('\nMô phỏng hoàn tất!\n');