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

int bad[905][31][31][4];
int dp[905][31][31][6][26];
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
void solve() {
    int n, m, K, B, S; cin >> n >> m >> K >> B >> S;
    vs a(n); cin >> a;
    int T = lcm(n, m);
    memset(bad, 0, sizeof(bad));
    memset(dp, -1, sizeof(dp));
    auto get = [](char ch) -> int {
        if(ch == 'U') return 0;
        if(ch == 'D') return 1;
        if(ch == 'L') return 2;
        return 3;
    };
    auto in = [&](int r, int c) -> int {
        return r >= 0 && r < n && c >= 0 && c < m;
    };
    while(K--) {
        int r, c; cin >> r >> c;
        char d; cin >> d;
        int v = get(d);
        for(int t = 0; t < T; t++) {
            int nr = r, nc = c;
            if(d == 'D') nr = r + t;
            else if(d == 'U') nr = r - t;
            else if(d == 'L') nc = c - t;
            else nc = c + t;
            nr = (nr % n + n) % n;
            nc = (nc % m + m) % m;
            bad[t][nr][nc][v] = 1;
        }
    }
    auto occ = [&](int g,int r,int c)->int{
        return bad[g][r][c][0]||bad[g][r][c][1]||bad[g][r][c][2]||bad[g][r][c][3];
    };
    auto get_other = [](int d) -> int {
        if(d == 0) return 1;
        if(d == 1) return 0;
        if(d == 2) return 3;
        return 2;
    };
    deque<ar(5)> q;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            if(a[i][j] == 'S') {
                if(!occ(0,i,j)) {
                    dp[0][i][j][B][0] = 0;
                    q.push_back({0,i,j,B,0});
                }
            }
        }
    }
    int FMAX = B*S;
    while(!q.empty()) {
        auto [g, r, c, b, s] = q.front(); q.pop_front();
        int t = dp[g][r][c][b][s];
        if(a[r][c] == 'E') {
            cout << t << '\n';
            return;
        }
        if(b > 0 && s < FMAX) {
            int nb = b - 1;
            int ns = min(FMAX, s + S);
            if(dp[g][r][c][nb][ns] == -1) {
                dp[g][r][c][nb][ns] = t;
                q.push_front({g, r, c, nb, ns});
            }
        }
        if(s > 0) {
            int g2 = g;
            int s2 = s - 1;
            if(!occ(g, r, c) && dp[g2][r][c][b][s2] == -1) {
                dp[g2][r][c][b][s2] = t + 1;
                q.push_back({g2, r, c, b, s2});
            }
            for(int j = 0; j < 4; j++) {
                int nr = r + dirs[j][0];
                int nc = c + dirs[j][1];
                if(!in(nr,nc) || a[nr][nc]=='X') continue;
                if(occ(g, nr, nc)) continue;
                if(dp[g2][nr][nc][b][s2] == -1) {
                    dp[g2][nr][nc][b][s2] = t + 1;
                    q.push_back({g2, nr, nc, b, s2});
                }
            }
        } else {
            int g2 = (g + 1) % T;
            if(!occ(g2, r, c) && dp[g2][r][c][b][0] == -1) {
                dp[g2][r][c][b][0] = t + 1;
                q.push_back({g2, r, c, b, 0});
            }
            for(int j = 0; j < 4; j++) {
                int nr = r + dirs[j][0];
                int nc = c + dirs[j][1];
                if(!in(nr,nc) || a[nr][nc]=='X') continue;
                if(occ(g2, nr, nc)) continue;
                int od = get_other(j);
                if(bad[g][nr][nc][od]) continue;
                if(dp[g2][nr][nc][b][0] == -1) {
                    dp[g2][nr][nc][b][0] = t + 1;
                    q.push_back({g2, nr, nc, b, 0});
                }
            }
        }
    }
    cout << -1 << '\n';
}

signed main() {
    IOS;
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        cout << "Case #" << i << ": ";  
        solve();
        if(i < t) {
            string v; cin >> v;
        }
    }
    return 0;
}

