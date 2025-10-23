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

if __name__ == "__main__":
    for k in range(3,13):
        m = 1000
        # make the directory if it does not exist
        import os
        if not os.path.exists(f"data/L{k}"):
            os.makedirs(f"data/L{k}")
        for l in range(0, 20):
            # bin [50l+1, 50(l+1)]
            s = 50*l+1
            r = 50*(l+1)
            strings_src, strings_tgt = generate_strings(m, max(s,k), r, k)
            with open(f"data/L{k}/{s}_to_{r}_src.txt", "w") as f:
                for string in strings_src:
                    f.write(string + "\n")

            with open(f"data/L{k}/{s}_to_{r}_tgt.txt", "w") as f:
                for string in strings_tgt:
                    f.write(string + "\n")
