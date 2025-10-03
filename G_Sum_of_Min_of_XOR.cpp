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

const int K = 30;
ll pre[K], dp[K][2];
int m;
ll count() {
    auto dfs = [&](auto dfs, int b = K - 1, int t = 1) -> ll {
        if(b == -1) return 1; 
        auto& res = dp[b][t];
        if(res != -1) return res;
        res = 0;
        int high = t ? (m >> b & 1) : 1;
        for(int d = 0; d <= high; d++) {
            if(pre[b] == -1 || pre[b] == d) {
                res += dfs(dfs, b - 1, t && d == high);
            }
        }
        return res;
    };
    memset(dp, -1, sizeof(dp));
    return dfs(dfs);
}
class Binary_Trie { 
    struct Node {
        int c[2];
        int cnt;
        Node() {
            c[0] = c[1] = 0;
            cnt = 0;
        }
    };
    public:
    vt<Node> T; // careful with static if no merging needed
    int root;
    int BIT;
    Binary_Trie(int _BIT = K) : BIT(_BIT){ root = new_node(); }

    int new_node() {
        T.pb(Node());
        return T.size() - 1;
    }
    
    void insert(ll num, int v = 1) {  
        int curr = root;   
        for(int i = BIT - 1; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(!T[curr].c[bits]) {
                T[curr].c[bits] = new_node();
            }
            curr = T[curr].c[bits];
            T[curr].cnt += v;
        }
    }

    ll ans = 0;
    void dfs(int b, int curr) {
        if(b < 0 || !count()) return;
        int l = T[curr].c[0];
        int r = T[curr].c[1];
        if(!T[l].cnt) {
            pre[b] = 0;
            ans += count() * (1LL << b);
            pre[b] = -1;
            dfs(b - 1, r);
        } else if(!T[r].cnt) {
            pre[b] = 1;
            ans += count() * (1LL << b);
            pre[b] = -1;
            dfs(b - 1, l);
            pre[b] = -1;
        } else {
            pre[b] = 0;
            dfs(b - 1, l);
            pre[b] = 1;
            dfs(b - 1, r);
            pre[b] = -1;
        }
    }
    
    ll get() {
        dfs(BIT - 1, root);
        return ans;
    }
};

void solve() {
    int n; cin >> n >> m;
    m--;
    vi a(n); cin >> a;
    srtU(a);
    Binary_Trie trie;
    memset(pre, -1, sizeof(pre));
    for(auto& x : a) {
        trie.insert(x);
    }
    cout << trie.get() << '\n';
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
