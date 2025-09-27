def compute(N,Z):
    n = 200000
    mask = (1 << 17) - 1
    fill = int((1 << 15) * 1.3 + 1)
    arr = [mask + 2] * 2
    x = 6
    for i in range(1, fill):
        arr += [x] + [x]
        x = x * 5 + 1
        x = x & mask

    arr += [1] * (n - len(arr))
    return arr


N = 2*10**5
max_value = 2*10**5
print(1)
print(N,2*10**5-1)
print(*compute(N,max_value))
for j in range(2*10**5-1):
    print(*[1,N-j])
