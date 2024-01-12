import os

for n in [10]:
    for cthr in [0.6, 0.7, 0.8]:
        for mthr in [0.6, 0.7, 0.8]:
            for region in ["guadeloupe"]:
                for rf in [
                    "none",
                    "h-east",
                    # "h-west",
                ]:
                    try:
                        os.system(
                            rfr"python mstme.py {cthr:.2f} {mthr:.2f} -r {region} -f {rf} --nbootstrap {n}"
                        )
                    except Exception as e:
                        print(e)

# for n in [2]:
#     for cthr in [0.8]:
#         for mthr in [0.8]:
#             for region in ["guadeloupe"]:
#                 for rf in [
#                     "h-east",
#                 ]:
#                     try:
#                         os.system(
#                             fr"python ./mstme.py {cthr:.2f} {mthr:.2f} -r {region} -f {rf} --nbootstrap {n}"
#                         )
#                     except Exception as e:
#                         print(e)
