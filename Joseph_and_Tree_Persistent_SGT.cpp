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
const static int MX = 1e5 + 5;
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
struct custom {
    static uint64_t splitmix64(uint64_t x) { x += 0x9e3779b97f4a7c15; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; return x ^ (x >> 31); }
    size_t operator()(uint64_t x) const { static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count(); return splitmix64(x + FIXED_RANDOM); }
};
    
const int MK = 22;
int root[MX * MK], T[MX], ptr, timer = 1, startTime[MX], endTime[MX], m;  
pii child[MX * MK];
vi d, st, cnt, tmp;   
vvpii graph;    
    
int dfs(int node, int par, int curr)   
{   
    d[node] = curr;
    st.pb(node);
    startTime[node] = timer++;  
    cnt[node] = 1;
    for(auto& [nei, w] : graph[node])
    {   
        if(nei != par) cnt[node] += dfs(nei, node, curr + w);    
    }   
    endTime[node] = timer - 1;  
    return cnt[node];
}
    
void update(int prev, int curr, int left, int right, int index) 
{   
    child[curr].ff = child[prev].ff;    
    child[curr].ss = child[prev].ss;
    if(left == right)   
    {   
        root[curr] = root[prev] + 1;
        return;
    }   
    int middle = left + (right - left) / 2; 
    if(index <= middle) 
    {   
        child[curr].ff = ++ptr; 
        update(child[prev].ff, child[curr].ff, left, middle, index);    
    }   
    else    
    {   
        child[curr].ss = ++ptr; 
        update(child[prev].ss, child[curr].ss, middle + 1, right, index);   
    }   
    root[curr] = root[child[curr].ff] + root[child[curr].ss];   
}

int queries(int prev, int curr, int left, int right, int k) 
{   
    if(left == right) return left; 
    int middle = left + (right - left) / 2; 
    int now = root[child[curr].ff] - root[child[prev].ff];  
    if(now >= k) return queries(child[prev].ff, child[curr].ff, left, middle, k);   
    return queries(child[prev].ss, child[curr].ss, middle + 1, right, k - now); 
}

int get(int v, int k)   
{   
    if(cnt[v] <= k) return -1;
    int index = queries(T[startTime[v]], T[endTime[v]], 0, m - 1, k);
    return tmp[index] - tmp[d[v]];
}
void solve()
{
    int n; cin >> n;    
    graph.rsz(n + 1), d.rsz(n + 1), cnt.rsz(n + 1);   
    for(int i = 0; i < n - 1; i++)  
    {   
        int a, b, w; cin >> a >> b >> w;    
        graph[a].pb({b, w});    
        graph[b].pb({a, w});    
    }   
    dfs(1, 0, 0);
    tmp = d;
    srtU(tmp);  
    m = tmp.size();
    umii mp;    
    for(int i = 0; i < tmp.size(); i++) mp[tmp[i]] = i; 
    for(auto& it : d) it = mp[it];  
    for(int i = 1; i <= st.size(); i++)  
    {   
        T[i] = ++ptr;   
        update(T[i - 1], T[i], 0, m - 1, d[st[i - 1]]);
    }
    int q; cin >> q;    
    while(q--)  
    {   
        int v, k; cin >> v >> k;
        cout << get(v, k) << endl;
    }
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
