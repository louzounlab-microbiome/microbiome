import spm1d


def hottelings_t2_test(Ya, Yb):
    T2 = spm1d.stats.hotellings2(Ya, Yb)
    T2i = T2.inference(0.05)
    print("p = {}, reject_h0 = {}".format(T2i.p, T2i.h0reject))