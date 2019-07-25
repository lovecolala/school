function gmf = cFuzzySet(data,parameter)
    im = sqrt(-1);
    center = parameter(1);
    sigma = parameter(2);
gmf = gauss_mf(data, [center sigma]).* exp(im*gauss_mf_d1_x(data, [center sigma]));