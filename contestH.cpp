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
#define vvpii vector<vpii>
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define var(x) vector<ar(x)>
#define pq priority_queue
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
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
const static int MX = 2e6 + 1;
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
    static const uint64_t C = 0x9e3779b97f4a7c15; const uint32_t RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const { return __builtin_bswap64((x ^ RANDOM) * C); }
    size_t operator()(const std::string& s) const { size_t hash = std::hash<std::string>{}(s); return hash ^ RANDOM; } };
template <class K, class V> using umap = std::unordered_map<K, V, custom>; template <class K> using uset = std::unordered_set<K, custom>;

struct Node 
{   
    int maxGap, cnt, leftGap, rightGap, range;
    Node() : leftGap(0), rightGap(0), cnt(0), maxGap(0), range(1) {}  
};  
int n;
vector<Node> root;
Node merge(Node left, Node right)   
{   
    Node res;   
    res.maxGap = max({left.maxGap, right.maxGap, left.rightGap + right.leftGap});   
    res.leftGap = left.leftGap + (left.leftGap == left.range ? right.leftGap : 0);
    res.rightGap = right.rightGap + (right.rightGap == right.range ? left.rightGap : 0);    
    res.range = left.range + right.range;   
    return res; 
}
    
void build(int i, int left, int right) 
{   
    if(left == right)   
    {   
        root[i].cnt = 0;
        root[i].maxGap = root[i].leftGap = root[i].rightGap = (root[i].cnt == 0);   
        return; 
    }   
    int middle = left + (right - left) / 2;
    build(i * 2 + 1, left, middle); 
    build(i * 2 + 2, middle + 1, right);    
    root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
}

void update(int id, int val, int i = 0, int left = 0, int right = MX - 1)   
{   
    if(left == right)   
    {   
        root[i].cnt += val; 
        root[i].maxGap = root[i].leftGap = root[i].rightGap = (root[i].cnt == 0);   
        root[i].range = 1;
        return; 
    }   
    int middle = left + (right - left) / 2; 
    if(id <= middle) update(id, val, i * 2 + 1, left, middle);  
    else update(id, val, i * 2 + 2, middle + 1, right); 
    root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);  
}   
    
int queries(int k, int i = 0, int left = 0, int right = MX - 1)
{   
    if(left == right) return left + 1;
    int middle = left + (right - left) / 2; 
    if(root[i * 2 + 1].maxGap >= k) return queries(k, i * 2 + 1, left, middle); 
    return queries(k, i * 2 + 2, middle + 1, right);
}   
    

void solve()
{
    root.assign(MX * 4, Node());
    build(0, 0, MX - 1);
    cin >> n;    
    for(int i = 0; i < n; i++)  
    {   
        int val; cin >> val;
        update(val, 1); 
    }
    int q; cin >> q;    
    while(q--)  
    {   
        char op; cin >> op; 
        if(op == '?')   
        {   
            int k; cin >> k;
            cout << queries(k) << " "; 
        }   
        else    
        {   
            int v; cin >> v;    
            update(v, (op == '+' ? 1 : -1));    
        }   
    }   
    cout << endl;


}

signed main()
{
    IOS;
    startClock

    int t = 1;
    cin >> t;
    while(t--) solve();

    endClock
    return 0;
}

