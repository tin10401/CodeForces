import sys

print(1)
for _ in range(1):
    n = 200000
    print (n,0,10**9)
    mask = (1<<17) - 1
    fill = int((1<<15)*1.3+1)
    arr = [mask+2]*2
    x = 6
    for i in range(1,fill):
        arr += [x]+[x]
        x = x * 5 + 1
        x = x & mask

    arr += [1]*(n-len(arr))
    #arr.pop()
    #arr = [2] + arr
    #print(*arr)
    new_arr = [arr[0]]
    for i in range(1,len(arr)):
        #curr_sum += arr[i]
        new_arr.append(arr[i]-arr[i-1])
    print(*[x for x in new_arr])
