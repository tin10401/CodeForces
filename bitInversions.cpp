// Author : Tin Le

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
#define vd vector<db>
#define ar(x) array<int, x>

#define pb push_back
#define f first
#define s second
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define rsrt(x) sort(allr(x))
#define sortErase(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define reverse(x) reverse(all(x))

const static int INF = 1LL << 61;
const static int MX = 2e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }

const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

constexpr int modExpo(int base, int exp, int mod)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

class SGT
{
    public:
    int n;
    vector<ar(5)> root;
    vi high, low;
    SGT(const string& s)
    {
        this->n = s.size();
        root.resize(n * 4);
        high.resize(n * 4), low.resize(n * 4);
        build(0, 0, n - 1, s);
    }
    bool allSame(int val, int i)
    {
        return val == high[i] - low[i] + 1;
    }

    void build(int i, int left, int right, const string &s)
    {
        low[i] = left, high[i] = right;
        if(left == right)
        {
            int bits = s[left] - '0';
            root[i] = {1, 1 - bits, 1 - bits, bits, bits};
            return;
        }
        int middle = left + (right - left) / 2;
        build(i * 2 + 1, left, middle, s);
        build(i * 2 + 2, middle + 1, right, s);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2], i);
    }

    ar(5) merge(ar(5) a, ar(5) b, int i)
    {
        ar(5) res = {};
        res[0] = max({a[0], b[0], a[2] + b[1], a[4] + b[3]});
        if(a[1]) res[1] = a[1] + (allSame(a[1], i * 2 + 1) ? b[1] : 0);
        else res[3] = a[3] + (allSame(a[3], i * 2 + 1) ? b[3] : 0);
        if(b[2]) res[2] = b[2] + (allSame(b[2], i * 2 + 2) ? a[2] : 0);
        else res[4] = b[4] + (allSame(b[4], i * 2 + 2) ? a[4] : 0);
        return res;
    }

    void update(int index)
    {
        update(0, 0, n - 1, index);
    }

    void update(int i, int left, int right, int index)
    {
        if(left == right)
        {
            int bits = root[i][1];
            root[i] = {1, 1 - bits, 1 - bits, bits, bits};
            return;
        }
        int middle = left + (right - left) / 2;
        if(index <= middle) update(i * 2 + 1, left, middle, index);
        else update(i * 2 + 2, middle + 1, right, index);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2], i);
    }

    int get()
    {
        return root[0][0];
    }
};



void solve()
{
    string s; cin >> s;
    SGT root(s);
    int n; cin >> n;
    while(n--)
    {
        int x; cin >> x;
        root.update(x - 1);
        cout << root.get() << " ";
    }
}

signed main()
{
    ios::sync_with_stdio(false); cin.tie(nullptr); 
    solve();
    return 0;
}

