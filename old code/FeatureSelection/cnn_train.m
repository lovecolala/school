function cnn = cnn_train(cnn, target, opts)
cnn.LearnCurve = [];
for epo = 1:opts.nEpoch
        %% FP
        cnn = cnn_fp(cnn, opts);
        
        %% BP
        cnn = cnn_bp(cnn, target, opts);
        
        cnn.LearnCurve = [cnn.LearnCurve, cnn.mse];
end
end