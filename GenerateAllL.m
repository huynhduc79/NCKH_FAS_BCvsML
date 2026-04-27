function Lset = GenerateAllL(N,B)
% Tạo toàn bộ vector L có dạng L1+...+LB = N

    if B == 1
        Lset = N;
        return;
    end

    Lset = [];
    for k = 1:N-B+1
        sub = GenerateAllL(N-k, B-1);
        Lset = [Lset; [k*ones(size(sub,1),1), sub]];
    end
end