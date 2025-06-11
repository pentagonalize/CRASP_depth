import random

def generate_ab(n,k):
    """
    L_1=a^+
    L_{k+1}=L_kb^+ if k odd, L_{k+1}=L_ka^+ if k even
    """

    # if k=1, return a^n
    if k == 1:
        return "a" * n
    else:
        # make a list of k-1 random numbers without repetition from 1 to n-1

        switching_indices = random.sample(range(1, n), k-1)
        switching_indices.sort()
        last_switch = switching_indices[-1]

        # string of 0's that is of length n
        string_tgt = "0" * last_switch + "1" * (n - last_switch)

        string_src="a"
        gen_a = True
        for i in range(1, n):
            if i in switching_indices:
                gen_a = not gen_a
            if gen_a:
                string_src += "a"
            else:
                string_src += "b"

        return string_src, string_tgt

# generate a list of m strings of lengths between n_1 and n_2 uniformly with k
def generate_strings(m, n_1, n_2, k):
    strings_src = []
    strings_tgt = []
    for i in range(m):
        n = random.randint(n_1, n_2)
        strings = generate_ab(n, k)
        strings_src.append(strings[0])
        strings_tgt.append(strings[1])

    return strings_src, strings_tgt

# test by generate a list of strings of length 10 with k=3
if __name__ == "__main__":
    # generate a corpus of 2k strings of length between 1 and 50 with k=3


    for k in range(3,13):
        m = 2000
        # make the directory if it does not exist
        import os
        if not os.path.exists(f"data/L{k}"):
            os.makedirs(f"data/L{k}")

        # bin [0, 50]
        strings_src, strings_tgt = generate_strings(m, k, 50, k)
        with open(f"data/L{k}/0_to_50_src.txt", "w") as f:
            for string in strings_src:
                f.write(string + "\n")

        with open(f"data/L{k}/0_to_50_tgt.txt", "w") as f:
            for string in strings_tgt:
                f.write(string + "\n")

        # bin [50, 100]
        strings_src, strings_tgt = generate_strings(m, 51, 100, k)
        with open(f"data/L{k}/50_to_100_src.txt", "w") as f:
            for string in strings_src:
                f.write(string + "\n")
        with open(f"data/L{k}/50_to_100_tgt.txt", "w") as f:
            for string in strings_tgt:
                f.write(string + "\n")

        # bin [100, 150]
        strings_src, strings_tgt = generate_strings(m, 101, 150, k)
        with open(f"data/L{k}/100_to_150_src.txt", "w") as f:
            for string in strings_src:
                f.write(string + "\n")
        with open(f"data/L{k}/100_to_150_tgt.txt", "w") as f:
            for string in strings_tgt:
                f.write(string + "\n")

        # bin [150, 200]
        strings_src, strings_tgt = generate_strings(m, 151, 200, k)
        with open(f"data/L{k}/150_to_200_src.txt", "w") as f:
            for string in strings_src:
                f.write(string + "\n")
        with open(f"data/L{k}/150_to_200_tgt.txt", "w") as f:
            for string in strings_tgt:
                f.write(string + "\n")

        # bin [200, 250]
        strings_src, strings_tgt = generate_strings(m, 201, 250, k)
        with open(f"data/L{k}/200_to_250_src.txt", "w") as f:
            for string in strings_src:
                f.write(string + "\n")
        with open(f"data/L{k}/200_to_250_tgt.txt", "w") as f:
            for string in strings_tgt:
                f.write(string + "\n")

        # bin [250, 300]
        strings_src, strings_tgt = generate_strings(m, 251, 300, k)
        with open(f"data/L{k}/250_to_300_src.txt", "w") as f:
            for string in strings_src:
                f.write(string + "\n")
        with open(f"data/L{k}/250_to_300_tgt.txt", "w") as f:
            for string in strings_tgt:
                f.write(string + "\n")
