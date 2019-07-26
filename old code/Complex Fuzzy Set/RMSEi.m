function rmse = RMSEi(error)
        %                         _______________
        %                       /   £U( error^2)
        %  RMSE =    /   --------------
        %                  ¡Ô              n
        
        sig_e2 = error * error';
        rmse = (sig_e2/length(error)).^0.5;
end