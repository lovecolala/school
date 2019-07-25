function [algoname, algofunc, algoopts] = algorithm(algotype)
switch algotype
        case 1
                algoname = 'PSO';
                algofunc =@ PSO;
                algoopts.nParticle = 30;
                algoopts.learning_rate = 1;
                algoopts.rlse = 0;
                
        case 2
                algoname = 'PSO-RLSE';
                algofunc =@ PSO;
                algoopts.nParticle = 30;
                algoopts.learning_rate = 1;
                algoopts.rlse = 1;
                
        case 3
                algoname = 'ABC';
                algofunc =@ ABC;
                algoopts.nBee = 30;
                algoopts.nOnlooker = 30;
                algoopts.limit = 10;
                algoopts.rlse = 0;
                
        case 4
                algoname = 'ABC-RLSE';
                algofunc =@ ABC;
                algoopts.nBee = 30;
                algoopts.nOnlooker = 30;
                algoopts.limit = 10;
                algoopts.rlse = 1;
                
        case 5
                algoname = 'SLPSO';
                algofunc =@ SLPSO;
                algoopts.nBaseParticle = 30;
                algoopts.rlse = 0;
                
        case 6
                algoname = 'SLPSO-RLSE';
                algofunc =@ SLPSO;
                algoopts.nBaseParticle = 30;
                algoopts.rlse = 1;
                
        case 7
                algoname = 'WOA';
                algofunc =@ WOA;
                algoopts.nAgent = 30;
                algoopts.rlse = 0;
                
        case 8
                algoname = 'WOA-RLSE';
                algofunc =@ WOA;
                algoopts.nAgent = 30;
                algoopts.rlse = 1;
                
        case 9
                algoname = 'CACO';
                algofunc =@ CACO;
                algoopts.nAnt = 5;
                algoopts.new_nAnt = 25;
                algoopts.eva_rate = 0.8;
                algoopts.learning_rate = 0.5;
                algoopts.rlse = 0;
                
        case 10
                algoname = 'CACO-RLSE';
                algofunc =@ CACO;
                algoopts.nAnt = 5;
                algoopts.new_nAnt = 25;
                algoopts.eva_rate = 0.8;
                algoopts.learning_rate = 0.5;
                algoopts.rlse = 1;
                
        otherwise
                error('You have a wrong algorithm name!');
                
end
end
