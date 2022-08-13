import os

for n in [50]:
    for thr in [1.00, 1.50, 2.00]:
        # for thr in [1.00, 1.25, 1.50, 1.75, 2.00]:
        for region in ['guadeloupe']:
            # for region in ['guadeloupe', 'caribbean']:
            for rf in [
                'none',
                'h-east',
                'h-west',
                'u-east',
                'u-west',
                # 'h-north',
                # 'h-south',
                # 'u-north',
                # 'u-south',
            ]:
                try:
                    os.system(
                        f"python mstme.py {thr:.2f} 0.25 -r {region} -f {rf} --nbootstrap {n}")
                except Exception as e:
                    print(e)
