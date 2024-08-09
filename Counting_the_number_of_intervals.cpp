// Author : Tin Le
//   __________________
//  | ________________ |
//  ||          ____  ||
//  ||   /\    |      ||
//  ||  /__\   |      ||
//  || /    \  |____  ||
//  ||________________||
//  |__________________|
//  \###################\
//   \###################\
//    \        ____       \
//     \_______\___\_______\
// An AC a day keeps the doctor away.

#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ll unsigned long long
#define int long long
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define vs vector<string>
#define vb vector<bool>
#define us unordered_set
#define um unordered_map
#define vvpii vector<vpii>
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define var(x) vector<ar(x)>
#define pq priority_queue
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define umii um<int, int, custom>
#define usi us<int, custom>
#define ff first
#define ss second
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#ifdef LOCAL
#define startClock clock_t tStart = clock();
#define endClock cout << fixed << setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
#else
#define startClock
#define endClock
#endif
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const static int INF = 1LL << 61;
const static int MX = 2e6 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vvi dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
struct custom {
    static uint64_t splitmix64(uint64_t x) { x += 0x9e3779b97f4a7c15; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; return x ^ (x >> 31); }
    size_t operator()(uint64_t x) const { static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count(); return splitmix64(x + FIXED_RANDOM); }
};

class FenwickTree   
{   
    public: 
    int n;  
    vi root;    
    FenwickTree(int n)  
    {   
        this->n = n;    
        root.rsz(n * 4);    
    }   
    
    void update(int i, int val) 
    {   
        while(i <= n)   
        {   
            root[i] += val; 
            i += (i & -i);  
        }   
    }
    
    int get(int i)  
    {   
        int res = 0;    
        while(i)    
        {   
            res += root[i]; 
            i -= (i & -i);  
        }   
        return res; 
    }   
};
void solve()
{
    int n, k; cin >> n >> k;    
    vi arr(n);  
    for(auto& it : arr) cin >> it;  
    vi tmp(arr);    
    srtU(tmp);  
    vi r(n), l(n);  
    iota(all(r), 0), iota(all(l), 0);   
    for(int i = 1; i < n; i++)
    {   
        int ptr = i;    
        while(ptr && arr[i] < arr[ptr - 1]) ptr = l[ptr - 1]; 
        l[i] = ptr;
    }   
    for(int i = n - 2; i >= 0; i--)    
    {   
        int ptr = i;    
        while(ptr + 1 < n && arr[i] <= arr[ptr + 1]) ptr = r[ptr + 1];  
        r[i] = ptr; 
    }   
    vvpii g(n); 
    for(int i = 0; i < n; i++)  
    {   
        if(i - l[i] <= r[i] - i)   
        {   
            for(int j = l[i]; j <= i; j++)  
            {   
                int d = ub(all(tmp), k - arr[i] - arr[j]) - begin(tmp); 
                g[r[i]].pb({d, 1}); 
                if(i) g[i - 1].pb({d, -1}); 
            }   
        }   
        else    
        {   
            for(int j = i; j <= r[i]; j++)  
            {   
                int d = ub(all(tmp), k - arr[i] - arr[j]) - begin(tmp); 
                g[i].pb({d, 1});    
                if(l[i]) g[l[i] - 1].pb({d, -1});   
            }   
        }   
    }   
    
    int res = 0;    
    FenwickTree root(n);
    for(int i = 0; i < n; i++)
    {   
        int d = lb(all(tmp), arr[i]) - begin(tmp);  
        root.update(d + 1, 1);  
        for(auto& [dis, v] : g[i])
        {   
            res += root.get(dis) * v;   
        }   
    }   
    cout << res << endl;
    


}

signed main()
{
    IOS;
    startClock

    int t = 1;
    // cin >> t;
    while(t--) solve();

    endClock
    return 0;
}

