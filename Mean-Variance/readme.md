objective
    Maximize alpha * h - (h_buy + h_sell) * cost - hvar

constraints
	0 <= h <= 0.02
	0 <= h_buy, h_sell
	h = h0 + h_buy - h_sell
	hvar = (h.T * expo) * cov * (expo.T * h) = (h.T*expo*L)*(U*expo.T*h) = h.T * (expo * cov * expo.T) * h =||h*(expo*L)||^2
	sum(h) <= 1


已知 alpha, cost, h0, expo, cov

求解 h, h_buy, h_sell
