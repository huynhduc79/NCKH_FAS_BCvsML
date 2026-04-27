function dataset_generate(NumSamples)
    final_dataset = [];  

    addpath('Core/')
    
    for s = 1:NumSamples
        % Random tham số hệ thống
        W     = 1 + 2* rand();     % Chiều dài từ 1 đến 3 lambda
        U     = randi([1,3]);
        gamma_dB = -5 + 10 * rand(); 
        gamma = 10^(gamma_dB / 10);  
        N     = randi([20, 30]);
        B     = 12;              
        
        % d: Khoảng cách giữa các cổng
        % mu2: Hệ số tương quan tương đương (sinc^2 hoặc sinc tùy mô hình)
        d = W / (N - 1);
        mu_val = besselj(0, 2*pi*d); % Tính J0(2*pi*d)
        mu2 = mu_val^2; 
        
        allL = GenerateAllL(N, B);
        bestP = 1;
        bestL = zeros(1,B);
        
        for i = 1:size(allL,1)
            L = allL(i,:);
            % Tính toán Outage tối ưu dựa trên mu2
            pout = CalcOutage(gamma, L, mu2, U, 'Quadrature', 20);
            if pout < bestP
                bestP = pout;
                bestL = L;
            end
        end
        
        final_dataset = [final_dataset; W U gamma N bestL]; 
        fprintf("Sample %d/%d DONE\n", s, NumSamples);
    end
    save dataset_final.mat final_dataset
    disp("Dataset tạo xong!");
end