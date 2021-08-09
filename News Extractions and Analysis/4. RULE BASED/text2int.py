def text2int(textnum, numwords={}):
    #if isinstance(int(textnum),int):
    if textnum=="orang": return 1;
    if textnum.isdigit():
        return int(textnum)
    if not numwords:
        units = [ "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan","sembilan"]
        scales = ["puluh", "ratus", "ribu", "juta", "miliar", "triliun"]
        se_belas = ["sepuluh","sebelas"]
        scale2 = ["belas","seratus"]
    
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(se_belas): numwords[word] = (1,10+(idx))
        for idx, word in enumerate(scale2): numwords[word] = (1,10**(idx+1))
        for idx, word in enumerate(scales):   
            if idx<2:
                numwords[word] = (10**(idx+1),0)
            else:
                numwords[word] = (10**((idx-1)*3),0)
                
    VISITED = VISITED2 = FOUND_PULUH = False
    current = result = curr_sem = 0
    for word in textnum.split():
        if word not in numwords:
            raise Exception("Illegal word: " + word)
        scale, increment = numwords[word]

        #mendeteksi puluhan
        if (current>=100 and not FOUND_PULUH) or (current>=100 and not VISITED):
            VISITED = True
            curr_sem = curr_sem * scale + increment
            if word in ["puluh","belas","sepuluh"] : FOUND_PULUH=True;
            else: continue
        if FOUND_PULUH and not VISITED2:
            VISITED2 = True
            current = current + curr_sem
            continue
        
        current = current * scale + increment
        if scale > 100:
            VISITED = VISITED2 = FOUND_PULUH =  False; curr_sem = 0;
            result += current
            current = 0
    return result+current