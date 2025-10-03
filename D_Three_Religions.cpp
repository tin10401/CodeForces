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

const int B = 255;
int dp[B][B][B];
void solve() {
    int n, q; cin >> n >> q;
    string t; cin >> t;
    t = ' ' + t;
    const int C = 26;
    vvi next(n + 1);
    {
        vi curr(C, inf);
        for(int i = n; i >= 0; i--) {
            next[i] = curr;
            if(i == 0) break;
            curr[t[i] - 'a'] = i;
        }
    }
    for(auto& x : dp) {
        for(auto& y : x) {
            for(auto& z : y) z = inf;
        }
    }
    dp[0][0][0] = 0;
    vvi s(3);
    auto update = [&](int id) -> void {
        const int N = s[0].size(), M = s[1].size(), K = s[2].size();
        for(int l1 = 0; l1 <= (id == 0 ? M : N); l1++) {
            for(int l2 = 0; l2 <= (id == 2 ? M : K); l2++) {
                int i, j, k;
                if(id == 0) i = N, j = l1, k = l2;
                else if(id == 1) i = l1, j = M, k = l2;
                else i = l1, j = l2, k = K;
                auto& res = dp[i][j][k];
                res = inf;
                if(i && dp[i - 1][j][k] <= n) {
                    res = min(res, next[dp[i - 1][j][k]][s[0][i - 1]]);
                }
                if(j && dp[i][j - 1][k] <= n) {
                    res = min(res, next[dp[i][j - 1][k]][s[1][j - 1]]);
                }
                if(k && dp[i][j][k - 1] <= n) {
                    res = min(res, next[dp[i][j][k - 1]][s[2][k - 1]]);
                }
            }
        }
    };
    while(q--) {
        char op; cin >> op;
        if(op == '+') {
            int i; cin >> i;
            char c; cin >> c;
            i--;
            int x = c - 'a';
            s[i].pb(x);
            update(i);
        } else {
            int i; cin >> i;
            i--;
            s[i].pop_back();
        }
        auto p = dp[s[0].size()][s[1].size()][s[2].size()];
        cout << (p <= n ? "YES" : "NO") << '\n';
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
