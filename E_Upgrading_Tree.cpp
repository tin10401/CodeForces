#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define ull unsigned long long
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

void solve() {
    int n; cin >> n;
    vvi graph(n + 1);
    for(int i = 1; i < n; i++) {
        int u, v; cin >> u >> v;
        graph[u].pb(v);
        graph[v].pb(u);
    }
    vi cent;
    {
        auto dfs = [&](auto& dfs, int node = 1, int par = 0) -> int {
            int s = 1;
            int mx = 0;
            for(auto& nei : graph[node]) {
                if(nei == par) continue;
                auto t = dfs(dfs, nei, node);
                mx = max(mx, t);
                s += t;
            }
            mx = max(mx, n - s);
            if(mx <= n / 2) {
                cent.pb(node);
            }
            return s;
        };
        dfs(dfs);
    }
    if(cent.empty()) {
        cout << 0 << '\n';
        return;
    }
    var(3) ops;
    auto upd = [&](int u, int v, int vv) -> void {
        if(v == vv) return;
        ops.pb({u, v, vv});
    };
    {
        auto process = [&](int c, int other) -> void {
            if(c == -1) return;
            for(auto& rt : graph[c]) {
                if(rt == other) continue;
                int last = rt;
                auto dfs = [&](auto& dfs, int u = 0, int par = -1) -> void {
                    for(auto& v : graph[u]) {
                        if(v == par || v == c || v == rt) continue;
                        upd(c, last, v);
                        upd(v, u, rt);
                        last = v;
                        dfs(dfs, v, u);
                    }
                };
                dfs(dfs, rt, -1);
                upd(c, last, rt);
            }
        };
        int c1 = cent[0];
        int c2 = (int)cent.size() > 1 ? cent[1] : -1;
        process(c1, c2);
        process(c2, c1);
    }
    cout << ops.size() << '\n';
    for(auto& [a, b, c] : ops) {
        cout << a << ' ' << b << ' ' << c << '\n';
    }
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
