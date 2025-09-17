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
const static ll INF = 1e18;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

int dp[5005][1 << 8];
void solve() {
    int n, k; cin >> n >> k;
    vi p(n + 1, n);
    for(int i = 0; i < n; i++) {
        int x; cin >> x;
        x--;
        p[x] = i;
    }
    for(int i = 0; i <= n; i++) {
        for(int j = 0; j < 1 << k; j++) {
            dp[i][j] = inf;
        }
    }
    vi prefix(n), curr(n);
    auto count = [&](int mn, int mask, int x) -> int {
        int res = prefix.back() - prefix[p[x]];
        for(int i = 0; i < k; i++) {
            if(mask >> i & 1) {
                if(p[x] < p[mn + i + 1]) res++;
            } 
        }
        return res;
    };
    dp[0][0] = 0;
    for(int mn = 0; mn < n; mn++) {
        prefix = curr;
        partial_sum(all(prefix), begin(prefix));
        int t = min(k, n - mn - 1);
        for(int mask = 0; mask < 1 << t; mask++) {
            for(int j = 0; j < t; j++) {
                if(mask >> j & 1) continue; 
                dp[mn][mask | (1 << j)] = min(dp[mn][mask | (1 << j)], dp[mn][mask] + count(mn, mask, mn + j + 1));
            } 
            int mn2 = mn + 1, mask2 = mask;
            while(mask2 & 1) mask2 >>= 1, mn2++;
            mask2 >>= 1;
            dp[mn2][mask2] = min(dp[mn2][mask2], dp[mn][mask] + count(mn, mask, mn));
        }
        curr[p[mn]] = 1;
    }
    cout << dp[n][0] << '\n';
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
