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

class SGT
{
    public:
    int n;
    var(2) root;
    vi lazy;
    SGT(vpii& arr)
    {
        n = arr.size();
        root.rsz(n * 4), lazy.rsz(n * 4);
        build(0, 0, n - 1, arr);
    }

    void apply(int i, int left, int right)
    {
        int& x = lazy[i];
        root[i][0] += x;
        root[i][1] += x;
        if(left != right)
        {
            lazy[i * 2 + 1] += lazy[i];
            lazy[i * 2 + 2] += lazy[i];
        }
        lazy[i] = 0;
    }

    void build(int i, int left, int right, vpii& arr)
    {
        if(left == right)
        {
            root[i][0] = root[i][1] = arr[left].ff;
            return ;
        }
        int middle = left + (right - left) / 2;
        build(i * 2 + 1, left, middle, arr);
        build(i * 2 + 2, middle + 1, right, arr);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    ar(2) merge(ar(2) left, ar(2) right)
    {
        ar(2) res;
        res[0] = min(left[0], right[0]);
        res[1] = max(left[1], right[1]);
        return res;
    }

    void update(int x)
    {
        update(0, 0, n - 1, x, -1);
    }

    void update(int i, int left, int right, int x, int val)
    {
        apply(i, left, right);
        if(root[i][0] > x)
        {
            lazy[i] += val;
            apply(i, left, right);
            return;
        }
        if(left == right || root[i][1] <= x) return;
        int middle = left + (right - left) / 2;
        update(i * 2 + 1, left, middle, x, val);
        update(i * 2 + 2, middle + 1, right, x, val);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    int get(int index)
    {
        return get(0, 0, n - 1, index);
    }

    int get(int i, int left, int right, int index)
    {
        apply(i, left, right);
        if(left == right) return root[i][0];
        int middle = left + (right - left) / 2;
        if(index <= middle) return get(i * 2 + 1, left, middle, index);
        return get(i * 2 + 2, middle + 1, right, index);
    }
};

void solve()
{
    int n; cin >> n;
    vpii arr(n);
    for(int i = 0; i < n; i++)
    {
        cin >> arr[i].ff;
        arr[i].ss = i;
    }
    srt(arr);
    SGT root(arr);
    int q; cin >> q;
    while(q--)
    {
        int x; cin >> x;
        root.update(x);
    }
    for(int i = 0; i < n; i++)
    {
        arr[i].ff = root.get(i);
        swap(arr[i].ff, arr[i].ss);
    }
    srt(arr);
    for(auto& it : arr) cout << it.ss << " ";
    cout << endl;

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

