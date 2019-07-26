clc; clear; close all;
current_path = [pwd, '\'];
record_path = [current_path, 'record\result_1021\'];
tt = 1;
RMSE = [];
save([record_path, 'parameter'], 'tt');
save([record_path, 'result_RMSE'], 'RMSE');
while tt <= 10
        % execute the main code
        FS_CNN_SCNFS_ver2;
        load parameter result_RMSE;
        
        str = ['result_NFS_', int2str(tt)];
        save([record_path, str]);
        
        RMSE = [RMSE; result_RMSE];
        tt = tt + 1;
        save([record_path, 'parameter'], 'tt');
        save([record_path, 'result_RMSE'], 'RMSE');
end