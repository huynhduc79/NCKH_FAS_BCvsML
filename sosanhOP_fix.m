%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% So sánh Pout: Jakes, Block-Diagonal (BC), và ML-Based Block-Diagonal
% W được cố định bằng 5.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('Core/') % Đảm bảo thư mục Core chứa SimOutage và CalcOutage

%-------------------------------------------------------------------------
% 1. Load mô hình Deep Learning đã train
%-------------------------------------------------------------------------
if exist('trained_net_softmax.mat', 'file')
    load('trained_net_softmax.mat'); % net
else
    error('Không tìm thấy file trained_net_softmax.mat!');
end

%-------------------------------------------------------------------------
% 2. Tham số hệ thống
%-------------------------------------------------------------------------
Nd = [2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]; 
W = 5;                       
U = [3 4 5];                 
Nsamples = 1e5;              
gam = 1;                     
gam_dB = 10*log10(gam);      
B = 12;                      % Số lượng khối cố định của ML

% Pre-allocation
pout_jakes  = zeros(length(U), length(Nd));
pout_blocks = zeros(length(U), length(Nd));
pout_ML     = zeros(length(U), length(Nd));


%-------------------------------------------------------------------------
% 3. Vòng lặp tính toán
%-------------------------------------------------------------------------
for ksize = 1:length(Nd)
    N = W * Nd(ksize);
    disp(['Đang xử lý N = ' num2str(N) ' (N/W = ' num2str(Nd(ksize)) ')']);
    
    Sigma_jakes = toeplitz(besselj(0, 2*pi*(0:N-1)*W/(N-1)));
    rho = sort(eig(Sigma_jakes), 'descend');
   
    d = W / (N - 1);
        mu_val = besselj(0, 2*pi*d); % Tính J0(2*pi*d)
        mu2 = mu_val^2; 
    mu2bc = mu2;

    Num_eig = sum(rho > N/100);
    L_BC = BlockCorrelation(N, rho, Num_eig, mu2bc); 
    
    for ku = 1:length(U)
        % --- A. Jakes's model ---
        pout_jakes(ku, ksize) = SimOutage(Nsamples, gam, Sigma_jakes, U(ku)); 
    
        % --- B. Block-Diagonal truyền thống ---
        pout_blocks(ku, ksize) = CalcOutage(gam, L_BC, mu2bc, U(ku), 'Quadrature', 25);
        
        % --- C. ML-Based Block-Diagonal ---
        input_ML = [W, U(ku), gam, N];
        P_predicted = predict(net, input_ML); 
        L_ML = round(P_predicted * N);
        
        % XỬ LÝ LỖI INFINITE LOOP: Ràng buộc thông minh
        if N >= B
            L_ML(L_ML < 1) = 1; % Nếu đủ cổng, ép tối thiểu 1
        else
            L_ML(L_ML < 0) = 0; % Nếu thiếu cổng, chỉ ép không được âm
        end
        
        % Bù trừ sai số phép làm tròn
        while sum(L_ML) ~= N
            diff_N = N - sum(L_ML);
            
            if diff_N > 0
                [~, max_idx] = max(L_ML);
                L_ML(max_idx) = L_ML(max_idx) + 1;
            elseif diff_N < 0
                if N >= B
                    valid_indices = find(L_ML > 1);
                else
                    valid_indices = find(L_ML > 0);
                end
                
                % Break khẩn cấp nếu không tìm được (Chống infinite loop tuyệt đối)
                if isempty(valid_indices)
                    break;
                end
                
                [~, max_idx_in_valid] = max(L_ML(valid_indices));
                real_idx = valid_indices(max_idx_in_valid);
                L_ML(real_idx) = L_ML(real_idx) - 1;
            end
        end
        
        % LỌC BỎ CÁC KHỐI CÓ 0 CỔNG TRƯỚC KHI TÍNH TOÁN
        L_ML = L_ML(L_ML > 0);
        
        pout_ML(ku, ksize) = CalcOutage(gam, L_ML, mu2, U(ku), 'Quadrature', 25);
    end
end


%-------------------------------------------------------------------------
% 4. Vẽ đồ thị
%-------------------------------------------------------------------------
figure('Color', 'w');
colors = ['r', 'b', 'g']; 
markers = ['*', 'o', 's'];
for ku = 1:length(U)
    semilogy(Nd, pout_jakes(ku,:), [colors(ku) markers(1) '-'], 'LineWidth', 1.5); hold on;
    semilogy(Nd, pout_blocks(ku,:), [colors(ku) markers(2) '--'], 'LineWidth', 1.5);
    semilogy(Nd, pout_ML(ku,:), [colors(ku) markers(3) ':'], 'LineWidth', 2);
end
grid on;
l = legend("Jake's (U=1)", "BC (U=1)", "ML-Based (U=1)", ...
           "Jake's (U=2)", "BC (U=2)", "ML-Based (U=2)", ...
           "Jake's (U=3)", "BC (U=3)", "ML-Based (U=3)");
set(l, 'FontSize', 12, 'Interpreter', 'latex', 'NumColumns', 3);
xlabel('$N/W$', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('$P_{out}(\gamma)$', 'FontSize', 14, 'Interpreter', 'latex');
title(['Saturation of FAS (W = ' num2str(W) ')'], 'Interpreter', 'latex');
ylim([1e-4, 1]);