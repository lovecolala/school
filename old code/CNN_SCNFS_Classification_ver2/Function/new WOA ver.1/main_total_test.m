%_________________________________________________________________________%
%  Whale Optimization Algorithm (WOA) source codes demo 1.0               %
%                                                                         %
%  Developed in MATLAB R2011b(7.13)                                       %
%                                                                         %
%  Author and programmer: Seyedali Mirjalili                              %
%                                                                         %
%         e-Mail: ali.mirjalili@gmail.com                                 %
%                 seyedali.mirjalili@griffithuni.edu.au                   %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                                                                         %
%   Main paper: S. Mirjalili, A. Lewis                                    %
%               The Whale Optimization Algorithm,                         %
%               Advances in Engineering Software , in press,              %
%               DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008   %
%                                                                         %
%_________________________________________________________________________%

% You can simply define your cost in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run WOA: [Best_score,Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%__________________________________________

close all;
clear;
clc;

SearchAgents_no=30; % Number of search agents
Max_iteration=1000; % Maximum numbef of iterations

Max_test = 30; % Maximun number of test

Save_result = false;
Save_figure = false;
Show_figure = false;
Show_result_each_round = true;

Current_path = pwd;
Save_path = strcat(Current_path, "/Data/");

for fun_id = 22:22
    Function_name = strcat('F', num2str(fun_id)); % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

    % Load details of the selected benchmark function
    [lb,ub,dim,fobj,bestSolution] = Get_Functions_details(Function_name);
    % dim = 30
    % Initialize the same positions for PSO & WOA
    Positions = initialization(SearchAgents_no, dim, ub, lb);

    % Andy
    disp(['Function name:', Function_name]);
    disp(['Best solution:', num2str(bestSolution)]);
    disp(['Save results:', num2str(Save_result)]);
    disp(['Search agents:', int2str(SearchAgents_no)]);
    disp(['Dimension:', int2str(dim)]);
    disp(['Iterations:', int2str(Max_iteration)]);
  
    for test_run=1:Max_test
        disp(['--------------- Run :', num2str(test_run),'/',num2str(Max_test), ' ---------------']);
        
        
        tic;
        [Best_score, Best_pos, new_WOA_cg_curve] = new_WOA(SearchAgents_no, Max_iteration, Positions, dim, ub, lb, fobj);
        new_WOA_result(test_run, 1) = test_run;
        new_WOA_result(test_run, 2) = Best_score;
        new_WOA_result(test_run, 3) = toc;
        new_WOA_Best_pos{test_run} = Best_pos;
        if Show_result_each_round == true
            disp(['The best optimal value of the objective funciton found by new_WOA is : ', num2str(Best_score)]); 
        end
        
        fig = figure('Position',[269   240   660   290]);
        %Draw search space
        subplot(1,2,1);

        func_plot(Function_name);
        title('Parameter space');
        grid on;
        xlabel('x_1');
        ylabel('x_2');
        zlabel([Function_name,'( x_1 , x_2 )']);

        %Draw objective space
        subplot(1,2,2);
        hold on;
        
        % Draw figure by plot() or semilogy.
        FlagSmallerThanZero_new_WOA = new_WOA_result(:, 2)<0;
        
        if sum(FlagSmallerThanZero_new_WOA) == 0
            semilogy(new_WOA_cg_curve, ':');
        else
            plot(new_WOA_cg_curve, ':');
        end
        
        hold off;
        title([Function_name, ' Objective space'])
        xlabel('Iteration');
        ylabel('Best score obtained so far');

        axis tight
        grid on
        box on
        legend('WOA-2(ver.4)');
       
        if Save_figure==true
            saveas(fig, strcat(Save_path, Function_name, '-other-times_', num2str(test_run), '.jpg'));
        end
        
        if Show_figure==false
            close(fig);
        end
    end % End of Andy
    
    if Save_result==true
        % Combine the results of different algorithm.
        Combined_result = [ new_WOA_result(:,2:3) ];
        % Calculate the average of result.
        if Max_test==1
            avg = Combined_result;
        else
            avg = mean(Combined_result);
        end
        Combined_result = [Combined_result; [0, avg(2:length(avg)) ] ];
        % Save raw data of this function.
        csvwrite(strcat(Save_path, Function_name, '_test_other.csv'), Combined_result);
        % Append summary result to the tail of test_summary.csv.
        dlmwrite(strcat(Save_path,'test_summary.csv'), [fun_id, bestSolution, test_run, dim, Max_iteration, avg(2:length(avg))],'delimiter',',','-append');
    end
    disp( '/////////////// Results ///////////////');
    disp([' Function name: ', Function_name]);
    disp([' Best solution: ', num2str(bestSolution)]);
    disp( ' - - - - - - - - - - - - - - - - - - -');
    disp(['   new WOA-2   Average: ', num2str(mean(new_WOA_result(:,2)))]);
    disp( '///////////////////////////////////////');
    disp(' ');
end
disp('File: main3_total_test.m');
