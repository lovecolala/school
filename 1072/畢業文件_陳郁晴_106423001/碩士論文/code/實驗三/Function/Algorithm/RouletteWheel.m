function l = RouletteWheel(prob)
Accum = cumsum(prob);           % Accumulate
l = find(rand<=Accum, 1);