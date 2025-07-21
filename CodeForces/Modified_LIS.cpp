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
const static int MX = 5e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vvi dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
void multiply(int f[2][2], int m[2][2]) {   
    int res[2][2] = {}; 
    for(int i = 0; i < 2; i++)  {   for(int j = 0; j < 2; j++)  {   for(int k = 0; k < 2; k++)  {   res[i][j] = (res[i][j] + f[i][k] * m[k][j]) % MOD; }   }   }   
    for(int i = 0; i < 2; i++)  {   for(int j = 0; j < 2; j++) f[i][j] = res[i][j]; }   }
int fib(int n)  {       if(n == 0) return 0;        if(n == 1) return 1;    
    int f[2][2] = {{1, 1}, {1, 0}}; int res[2][2] = {{1, 0}, {0, 1}};       
    while(n)    {   if(n & 1) multiply(res, f); multiply(f, f); n >>= 1;    }   return res[0][1] % MOD; }   
int GCD[MX], TOTI[MX];  
void gcdSum()  {   for(int i = 0; i < MX; i++) TOTI[i] = i;   
    for(int i = 2; i < MX; i++) {   if(TOTI[i] == i)   {   TOTI[i] = i - 1; for(int j = 2 * i; j < MX; j += i)  {   TOTI[j] -= (TOTI[j] / i); }   }   }   
    for(int i = 1; i < MX; i++) {   for(int j = i, k = 1; j < MX; j += i, k++)  {   GCD[j] += i * TOTI[k];   }   }
}
struct custom {
    static uint64_t splitmix64(uint64_t x) { x += 0x9e3779b97f4a7c15; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; return x ^ (x >> 31); }
    size_t operator()(uint64_t x) const { static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count(); return splitmix64(x + FIXED_RANDOM); }
};
    
struct Node 
{   
    int way, mx;    
    Node() : way(0), mx(0) {}   
    Node(int w, int m) : way(w), mx(m) {}
};  
    
class SGT   
{   
    public:
    int n;  
    vector<Node> root;
    SGT(int n)  
    {   
        this->n = n;    
        root.rsz(n * 4);
    }   
    
    void update(int index, Node val)
    {   
        update(0, 0, n - 1, index, val);
    }   
    
    void update(int i, int left, int right, int index, Node val)
    {   
        if(left == right)   
        {   
            root[i] = merge(root[i], val);  
            return; 
        }   
        int middle = left + (right - left) / 2; 
        if(index <= middle) update(i * 2 + 1, left, middle, index, val);    
        else update(i * 2 + 2, middle + 1, right, index, val);  
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);  
    }   
    
    Node queries(int start, int end)    
    {   
        return queries(0, 0, n - 1, start, end);    
    }   
    
    Node merge(Node left, Node right)   
    {   
        Node res;   
        res.mx = max(left.mx, right.mx);    
        res.way = ((res.mx == left.mx ? left.way : 0) + (res.mx == right.mx ? right.way : 0)) % MOD;
        return res;
    }   

    Node queries(int i, int left, int right, int start, int end)
    {   
        if(left > end || start > right) return Node();  
        if(left >= start && right <= end) return root[i];   
        int middle = left + (right - left) / 2;
        return merge(queries(i * 2 + 1, left, middle, start, end), queries(i * 2 + 2, middle + 1, right, start, end));
    }   
};


void solve()
{
    int n; cin >> n;    
    vi arr(n);  
    int m = 0;  
    for(int i = 0; i < n; i++)  
    {   
        cin >> arr[i];  
        m = max(m, abs(arr[i])); 
    }   
    SGT neg(m + 1), pos(m + 1);
    for(auto& it : arr) 
    {   
        if(it < 0)  
        {   
            Node k = pos.queries(0, abs(it) - 1);   
            if(k.mx == 0) k.way = 1;    
            k.mx++; 
            neg.update(abs(it), k); 
        }   
        else    
        {   
            Node k = neg.queries(0, it - 1);    
            if(k.mx == 0) k.way = 1;    
            k.mx++; 
            pos.update(it, k);  
        }   
    }
    Node k = pos.merge(pos.queries(0, m), neg.queries(0, m));   
    cout << k.mx << " " << k.way << endl;
}

signed main()
{
    IOS;
    startClock

    int t = 1;
    //cin >> t;
    while(t--) solve();

    endClock
    return 0;
}

