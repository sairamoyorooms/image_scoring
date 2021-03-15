# Gradient_aes_model
def aes_grad(col):
    if col <= 1.00007:
        return 0 + ((col - 0) / (1.00007 - 0))
    if col > 1.00007 and col <= 1.32409:
        return 1 + ((col - 1.00007) / (1.00007 - 7.236282))
    if col > 1.32409 and col <= 2.95663:
        return 2 + ((col - 1.32409) / (2.95663 - 1.32409))
    if col > 2.95663 and col <= 4.76146:
        return 3 + ((col - 2.95663) / (4.76146 - 2.95663))
    if col > 4.76146 and col <= 6.59607:
        return 4 + ((col - 4.76146) / (6.59607 - 4.76146))
    if col > 5.99812 and col <= 6.59607:
        return 5 + ((col - 5.99812) / (6.59607 - 5.99812))
    if col > 6.59607 and col <= 7.10172:
        return 6 + ((col - 6.59607) / (7.10172 - 6.59607))
    if col > 7.10172 and col <= 7.82517:
        return 7 + ((col - 7.10172) / (7.82517 - 7.10172))
    if col > 7.82517 and col <= 8.00960:
        return 8 + ((col - 7.82517) / (8.00960 - 7.82517))
    if col > 8.00960:
        return 9 + ((col - 8.00960) / (10 - 8.00960))


# Gradient_tech_new_model
def tech_grad(col):
    if col <= 1.15888:
        return 0 + ((col - 0) / (1.15888 - 0))
    if col > 1.15888 and col <= 2.22530:
        return 1 + ((col - 1.15888) / (2.22530 - 1.15888))
    if col > 2.22530 and col <= 3.56137:
        return 2 + ((col - 2.22530) / (3.56137 - 2.22530))
    if col > 3.56137 and col <= 4.82128:
        return 3 + ((col - 3.56137) / (4.82128 - 3.56137))
    if col > 4.82128 and col <= 5.13893:
        return 4 + ((col - 4.82128) / (5.13893 - 4.82128))
    if col > 5.13893 and col <= 5.99385:
        return 5 + ((col - 5.13893) / (5.99385 - 5.13893))
    if col > 5.99385 and col <= 6.62869:
        return 6 + ((col - 5.99385) / (6.62869 - 5.99385))
    if col > 6.62869 and col <= 7.36670:
        return 7 + ((col - 6.62869) / (7.36670 - 6.62869))
    if col > 7.36670 and col <= 8.86024:
        return 8 + ((col - 7.36670) / (8.86024 - 7.36670))
    if col > 8.86024:
        return 9 + ((col - 8.86024) / (10 - 8.86024))