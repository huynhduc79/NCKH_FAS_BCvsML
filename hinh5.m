% Script so sánh thời gian chạy giữa BC và DL
N_test = 20:20:1000; % Thử với N từ 20 đến 2000
time_BC = zeros(length(N_test), 1);
time_DL = zeros(length(N_test), 1);

addpath('Core/')

W = 5; mu2 = 0.97;
load('trained_net_softmax.mat'); % Load mô hình DL

for i = 1:length(N_test)
    N = N_test(i);
    
    % --- ĐO THỜI GIAN BC ---
    tic;
    Sigma_jakes = toeplitz(besselj(0, 2*pi*(0:N-1)*W/(N-1)));
    rho = sort(eig(Sigma_jakes), 'descend');
    Num_eig = sum(rho > N/100);
    L_BC = BlockCorrelation(N, rho, Num_eig, mu2);
    time_BC(i) = toc;
    
    % --- ĐO THỜI GIAN DL ---
    tic;
    input_DL = [W, 2, 0, N]; % [W, U, gamma_dB, N]
    P_predicted = predict(net, input_DL); 
    L_DL = round(P_predicted * N);
    time_DL(i) = toc;
end

% Vẽ đồ thị
figure;
plot(N_test, time_BC, 'b-o', 'LineWidth', 2); hold on;
plot(N_test, time_DL, 'r-*', 'LineWidth', 2);
xlabel('Số lượng cổng (N)');
ylabel('Thời gian tính toán (giây)');
legend('Thuật toán BC (Toán học)', 'Mô hình DL (Dự đoán)');
title('So sánh độ phức tạp tính toán: BC vs DL');
grid on;