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
    int n, z; cin >> n >> z;
    vi a(n); cin >> a;
    const int m = log2(n) + 1;
    vvi dp1(n + 1, vi(m, n));
    for(int i = 0, j = 0; i < n; i++) {
        while(j < n && a[i] + z >= a[j]) j++;
        dp1[i][0] = j;
    }
    dp1[n][0] = n;
    vi depth(n + 1);
    for(int i = n - 1; i >= 0; i--) {
        depth[i] = depth[dp1[i][0]] + 1;
    }
    const int K = 2;
    for(int j = 1; j < m; j++) {
        for(int i = 0; i <= n; i++) {
            dp1[i][j] = dp1[dp1[i][j - 1]][j - 1];
        }
    }
    auto lca = [&](int u, int v) -> int {
        if(depth[u] < depth[v]) swap(u, v);
        int d = depth[u] - depth[v];
        for(int j = 0; j < m; j++) {
            if(d >> j & 1) u = dp1[u][j];
        }
        if(u == v) return u;
        for(int j = m - 1; j >= 0; j--) {
            if(dp1[u][j] != dp1[v][j]) {
                u = dp1[u][j];
                v = dp1[v][j];
            }
        }
        return dp1[u][0];
    };
    vvpii dpk(n + 1, vpii(m));
    dpk[n][0] = {n, 0};
    for(int i = 0; i < n; i++) {
        int t = i;
        int d = 0;
        for(int j = i + 1; j <= min(n, i + K - 1); j++) {
            t = lca(t, j);
        }
        for(int j = i; j <= min(n, i + K - 1); j++) {
            d += depth[j] - depth[t];
        }
        dpk[i][0] = {t, d};
    }
    for(int j = 1; j < m; j++) {
        for(int i = 0; i <= n; i++) {
            int p = dpk[i][j - 1].ff;
            dpk[i][j] = {dpk[p][j - 1].ff, dpk[i][j - 1].ss + dpk[p][j - 1].ss};
        }
    }
    auto jump = [&](int l, int r) -> int {
        if(l > r) return 0;
        int u = l;
        int res = 1;
        for(int j = m - 1; j >= 0; j--) {
            int p = dp1[u][j];
            if(p <= r) {
                res += 1LL << j;
                u = p;
            }
        }
        return res;
    };
    int q; cin >> q;
    while(q--) {
        int l, r; cin >> l >> r;
        l--, r--;
        int res = 0;
        for(int j = m - 1; j >= 0; j--) {
            int p = dpk[l][j].ff;
            if(p <= r) {
                res += dpk[l][j].ss;
                l = p;
            }
        }
        res += jump(l, r) + jump(l + 1, r);
        cout << res << '\n';
    }
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
