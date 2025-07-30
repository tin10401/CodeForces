struct Point {
    ld x, y;
    Point() : x(0), y(0) {}
    Point(ld x, ld y) : x(x), y(y) {}
    double distance(const Point &other) const {
        return std::sqrt((x - other.x) * (x - other.x) +
                         (y - other.y) * (y - other.y));
    }
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }

    ld slope_to(const Point &other) const {
        if (std::fabs(other.x - x) < eps) return std::numeric_limits<ld>::infinity();
        return (other.y - y) / (other.x - x);
    }
};

bool is_collinear(const Point &p1, const Point &p2, const Point &p3) {
    ld cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    return std::fabs(cross) < eps;
}

ld triangle_area(const Point &p1, const Point &p2, const Point &p3) {
    ld cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    return std::fabs(cross) / 2.0;
}

struct Circle {
    ld x, y, r;
    Circle(ld x = 0, ld y = 0, ld r = 0) : x(x), y(y), r(r) {}

    ld intersect(const Circle& o) const {
        // https://codeforces.com/contest/600/problem/D
        ld dx = x - o.x;
        ld dy = y - o.y;
        ld d  = sqrt(dx * dx + dy * dy);
        if(d >= r + o.r) return 0.0L;
        if(d <= fabsl(r - o.r)) {
            ld rr = min(r, o.r);
            return M_PI * rr * rr;
        }
        ld r2 = r * r, R2 = o.r * o.r, d2 = d * d;
        ld alpha = acosl((d2 + r2 - R2) / (2 * d * r)) * 2;
        ld beta = acosl((d2 + R2 - r2) / (2 * d * o.r)) * 2;
        ld area1 = 0.5L * r2 * (alpha - sinl(alpha));
        ld area2 = 0.5L * R2 * (beta - sinl(beta));
        return area1 + area2;
    }
};


struct Rectangle {
    ld x, y, w, h;
    Rectangle(ld x = 0, ld y = 0, ld w = 0, ld h = 0) : x(x), y(y), w(w), h(h) {}
};

bool circlesMergeWithinRect(const Circle &c1, const Circle &c2, const Rectangle &rect) { // check if the both circle intersect
                                                                                         // where the intersection is within the rectangle
    ld dx = c2.x - c1.x;
    ld dy = c2.y - c1.y;
    ld d = std::sqrt(dx * dx + dy * dy);
    if (d > c1.r + c2.r || d < fabsl(c1.r - c2.r)) return false;
    ld a = (c1.r * c1.r - c2.r * c2.r + d * d) / (2 * d);
    ld temp = c1.r * c1.r - a * a;
    if(temp < 0) temp = 0;
    ld h = std::sqrt(temp);
    ld px = c1.x + a * dx / d;
    ld py = c1.y + a * dy / d;
    if (fabs(h) < eps) {
        return (px >= rect.x && px <= rect.x + rect.w &&
                py >= rect.y && py <= rect.y + rect.h);
    } else {
        ld rx = -h * dy / d;
        ld ry = h * dx / d;
        ld ix1 = px + rx;
        ld iy1 = py + ry;
        ld ix2 = px - rx;
        ld iy2 = py - ry;
        bool inside1 = (ix1 >= rect.x && ix1 <= rect.x + rect.w &&
                        iy1 >= rect.y && iy1 <= rect.y + rect.h);
        bool inside2 = (ix2 >= rect.x && ix2 <= rect.x + rect.w &&
                        iy2 >= rect.y && iy2 <= rect.y + rect.h);
        return inside1 || inside2;
    }
}

bool circleRectangleIntersect(const Circle &c, const Rectangle &rect) { // check if circle intersect with the rectangle
    ld closestX = std::max(rect.x, std::min(c.x, rect.x + rect.w));
    ld closestY = std::max(rect.y, std::min(c.y, rect.y + rect.h));
    ld dx = c.x - closestX;
    ld dy = c.y - closestY;
    return (dx * dx + dy * dy) <= c.r * c.r;
}

bool circlesIntersect(const Circle &c1, const Circle &c2) { // check if two circle intersect
    ld dx = c1.x - c2.x;
    ld dy = c1.y - c2.y;
    ld distanceSq = dx * dx + dy * dy;
    ld radiusSum = c1.r + c2.r;
    return distanceSq <= radiusSum * radiusSum;
}

bool circleLineIntersect(const Circle &c, const Point &p1, const Point &p2) { // check if a circle intersect with a line
                                                                              // (x1, y1) to (x2, y2) inclusive
    long double cx = c.x, cy = c.y, cr = c.r;
    long double dx = p2.x - p1.x, dy = p2.y - p1.y;
    long double lenSq = dx * dx + dy * dy;
    if (lenSq == 0.0L) {
        long double distSq = (cx - p1.x) * (cx - p1.x) + (cy - p1.y) * (cy - p1.y);
        return distSq <= cr * cr;
    }
    long double t = ((cx - p1.x) * dx + (cy - p1.y) * dy) / lenSq;
    long double closestX, closestY;
    if (t < 0.0L) {
        closestX = p1.x;
        closestY = p1.y;
    } else if (t > 1.0L) {
        closestX = p2.x;
        closestY = p2.y;
    } else {
        closestX = p1.x + t * dx;
        closestY = p1.y + t * dy;
    }
    long double distSq = (cx - closestX) * (cx - closestX) + (cy - closestY) * (cy - closestY);
    return distSq <= cr * cr;
}

pll getMidpointKey(const pll& p1, const pll& p2) { // return the key to determine if two line are parallel
    return {p1.ff + p2.ff, p1.ss + p2.ss};
}

bool areParallelByMidpoint(const pll& p1, const pll& p2, const pll& q1, const pll& q2) {
    return getMidpointKey(p1,p2) == getMidpointKey(q1,q2);
}

struct manhattan {
    ll mx1 = -INF, mn1 = INF, mx2 = -INF, mn2 = INF;
    void update(ll x, ll y) {
        mx1 = max(mx1, x + y);
        mn1 = min(mn1, x + y);
        mx2 = max(mx2, x - y);
        mn2 = min(mn2, x - y);
    }

    ll query() { return max(mx1 - mn1, mx2 - mn2); }
};

struct Rect { ll x1, y1, x2, y2; };

struct SheetManager {
    // https://codeforces.com/contest/1216/problem/C
    vt<Rect> placed;
    bool insert(const Rect &r) {
        vt<Rect> clip;
        for (auto &q : placed) {
            Rect c{max(r.x1, q.x1), max(r.y1, q.y1), min(r.x2, q.x2), min(r.y2, q.y2)};
            if(c.x1 < c.x2 && c.y1 < c.y2) clip.push_back(c);
        }
        if(clip.empty()) {
            placed.pb(r);
            return true;
        }
        vll ys;
        ys.reserve(clip.size() * 2 + 2);
        ys.pb(r.y1);
        ys.pb(r.y2);
        for(auto &c : clip) {
            ys.pb(c.y1);
            ys.pb(c.y2);
        }
        srtU(ys);
        int Y = ys.size();
        struct Node { int cnt, len; };
        vt<Node> st(4 * Y);
        vt<tuple<int, int, int, int>> events;
        events.reserve(clip.size() * 2);
        for (auto &c : clip) {
            int y1 = lb(all(ys), c.y1) - ys.begin();
            int y2 = lb(all(ys), c.y2) - ys.begin();
            events.emplace_back(c.x1, y1, y2, 1);
            events.emplace_back(c.x2, y1, y2, -1);
        }
        sort(events.begin(), events.end(), [](auto &a, auto &b) { return get<0>(a) < get<0>(b); });
        auto update = [&](auto& update, int node, int l, int r, int ql, int qr, int v) {
            if(ql >= r || qr <= l) return;
            if(ql <= l && r <= qr) st[node].cnt += v;
            else {
                int m = (l + r) >> 1;
                update(update, node << 1, l, m, ql, qr, v);
                update(update, node << 1 | 1, m, r, ql, qr, v);
            }
            if(st[node].cnt > 0) st[node].len = ys[r] - ys[l];
            else if(r - l == 1) st[node].len = 0;
            else st[node].len = st[node << 1].len + st[node << 1 | 1].len;
        };
        ll covered = 0;
        int prevX = get<0>(events[0]);
        for(auto &e : events) {
            int x, y1, y2, tp;
            tie(x, y1, y2, tp) = e;
            covered += 1LL * (x - prevX) * st[1].len;
            update(update, 1, 0, Y - 1, y1, y2, tp);
            prevX = x;
        }
        ll total = 1LL * (r.x2 - r.x1) * (r.y2 - r.y1);
        if(covered < total) {
            placed.pb(r);
            return true;
        }
        return false;
    }
};

struct Line {
    mutable ll m, c, p;
    bool isQuery;
    bool operator<(const Line& o) const { if(o.isQuery) return p < o.p; return m < o.m; }
};

struct CHT : multiset<Line> { // cht max, for min just inverse the sign
    mutable iterator best;
    mutable bool init = false;

    ll div(ll a, ll b) { return a / b - ((a ^ b) < 0 && a % b); }

    bool isect(iterator x, iterator y) {
        if (y == end()) { x->p = INF; return false; }
        if (x->m == y->m) x->p = x->c > y->c ? INF : -INF;
        else x->p = div(y->c - x->c, x->m - y->m);
        return x->p >= y->p;
    }
	
//    inline ld div(const Line& X, const Line& Y) { // for ld comparison
//        return (Y.c - X.c) / (X.m - Y.m);
//    }
//
//    bool isect(iterator x, iterator y) {
//        if(y == end()) {
//            x->p = INF;
//            return false;
//        }
//        if(x->m == y->m) {
//            x->p = x->c > y->c ? INF : -INF;
//        } else {
//            x->p = div(*x, *y); 
//        }
//        return x->p >= y->p;
//    }

    void add(ll m, ll c) {
        auto z = insert({m, c, 0, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p) isect(x, erase(y));
    }

    ll query(ll x) {
        if(empty()) return -INF;
        Line q; q.p = x, q.isQuery = 1;
        auto l = *lower_bound(q);
        return l.m * x + l.c;
    }

    ll linear_query(ll x) {
        if(empty()) return -INF;
        if(!init) { best = begin(); init = true; }
        while(next(best) != end() && next(best)->m * x + next(best)->c >= best->m * x + best->c) best++;
        return best->m * x + best->c;
    }
};

class CHT_segtree { // max cht
public: 
    int n, base;
    vt<CHT> tree;

    CHT_segtree(int n) : n(n) {
        base = 1;
        while(base < n) base <<= 1;
        tree.rsz(base << 1);
    }
    
    void update_at(int id, pll val) {  
        if(id >= n) return;
        int pos = id + base;
        while(pos > 0) {
            tree[pos].add(val.ff, val.ss);
            pos >>= 1;
        }
    }

    ll queries_at(int id, ll x) {
        if(id < 0 || id >= n) return -INF;
        ll ans = -INF;
        int pos = id + base;
        while(pos > 0) {
            ans = max(ans, tree[pos].query(x));
            pos >>= 1;
        }
        return ans;
    }

    void update_range(int l, int r, pll val) { // be careful it doesn't add downward, so only use this when you queries_at
        if(l < 0) l = 0;
        if(r >= n) r = n - 1;
        if(l > r) return;
        int L = l + base, R = r + base;
        while(L <= R) {
            if(L & 1) tree[L++].add(val.ff, val.ss);
            if(!(R & 1)) tree[R--].add(val.ff, val.ss);
            L >>= 1; R >>= 1;
        }
    }

    ll queries_range(int l, int r, ll x) { 
        if(l < 0 || r >= n) return -INF;
        ll ans = -INF;
        l += base, r += base;
        while(l <= r) {
            if(l & 1) ans = max(ans, tree[l++].query(x));
            if(!(r & 1)) ans = max(ans, tree[r--].query(x)); 
//            if (l & 1) ans = max(ans, tree[l++].linear_query(x));
//            if (!(r & 1)) ans = max(ans, tree[r--].linear_query(x)); 
            l >>= 1, r >>= 1;
        }
        return ans;
    }
};

struct Undo_CHT { // ll version
    struct Line {
        ll a, b;   // y = a*x + b
    };
    struct UndoEntry {
        Line prev;
        ll pos, old_sz;
    };

    vt<Line> A;
    vt<UndoEntry> undo;
    ll sz = 0;

    Undo_CHT(int max_n) {
        A.resize(max_n);
        undo.reserve(max_n);
    }

    static bool bad(const Line &l1, const Line &l2, const Line &l3) {
        i128 lhs = i128(l2.b - l1.b) * (l2.a - l3.a);
        i128 rhs = i128(l3.b - l2.b) * (l1.a - l2.a);
        return lhs >= rhs;
    }

    void add(ll u, ll v) {
        Line x = {u, v};
        ll l = 1, r = sz - 1, ans = sz;
        while(l <= r) {
            ll mid = (l + r) / 2;
            if(bad(A[mid - 1], A[mid], x)) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        undo.pb({A[ans], ans, sz});
        A[ans] = x;
        sz = ans + 1;
    }

    void roll_back() {
        auto e = undo.back(); undo.pop_back();
        sz = e.old_sz;
        A[e.pos] = e.prev;
    }

    ll cal(const Line &L, ll x) const {
        return L.a * x + L.b;
    }

    ll query(ll x) const {
        if(sz == 0) return -INF;
        ll ans = cal(A[0], x);
        ll l = 1, r = sz - 1;
        while(l <= r) {
            ll mid = (l + r) / 2;
            ll y_mid = cal(A[mid], x);
            ll y_prev = cal(A[mid - 1], x);
            if(y_mid > y_prev) {
                ans = max(ans, y_mid);
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }
};

struct Undo_CHT { // ld version
    struct Line {
        ld a, b;   // y = a*x + b
    };
    struct UndoEntry {
        Line prev;
        int pos;
        int old_sz;
    };

    vt<Line> A;
    vt<UndoEntry> undo;
    int sz = 0;

    Undo_CHT(int maxn) {
        A.rsz(maxn);
        undo.reserve(maxn);
    }

    static bool bad(const Line &l1, const Line &l2, const Line &l3) {
        return (l2.b - l1.b) * (l2.a - l3.a) >= (l3.b - l2.b) * (l1.a - l2.a);
    }

    void add(ld u, ld v) {
        Line x{u, v};
        int l = 1, r = sz - 1, pos = sz;
        while(l <= r) {
            int mid = (l + r) >> 1;
            if(bad(A[mid - 1], A[mid], x)) {
                pos = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        undo.pb({A[pos], pos, sz});
        A[pos] = x;
        sz = pos + 1;
    }

    void roll_back() {
        auto e = undo.back(); undo.pop_back();
        sz = e.old_sz;
        A[e.pos] = e.prev;
    }

    ld cal(const Line &L, ld x) const {
        return L.a * x + L.b;
    }

    ld query(ld x) const {
        if (sz == 0) return -1e300; 
        ld res = cal(A[0], x);
        int l = 1, r = sz - 1;
        while(l <= r) {
            int mid = (l + r) >> 1;
            ld y1 = cal(A[mid - 1], x);
            ld y2 = cal(A[mid],   x);
            if(y2 > y1) {
                res = max(res, y2);
                l = mid + 1;
            } else {
                res = max(res, y1);
                r = mid - 1;
            }
        }
        return res;
    }
};

struct Line { ll m, b; ll eval(ll x) const { return m * x + b; } };
struct MonoCHT { // max cht for monotonic function(prefix sum with all positive, ...)
    deque<Line> dq;
    bool increasing_query;
    MonoCHT(bool _inc = true) : increasing_query(_inc) {}

    bool bad(const Line &L1, const Line &L2, const Line &L3) {
        auto L = (L3.b - L1.b) * (L1.m - L2.m);
        auto R = (L2.b - L1.b) * (L1.m - L3.m);
        return increasing_query ? L <= R : L >= R;
    }
    void add(Line L) {
        while(dq.size() >= 2 && bad(dq[dq.size() - 2], dq[dq.size() - 1], L)) dq.pop_back();
        dq.pb(L);
    }
    ll query(ll x) {
        while(dq.size() >= 2 && dq[0].eval(x) <= dq[1].eval(x)) dq.pop_front();
        return dq[0].eval(x);
    }
};

template<typename T>
struct LiChao {
    static const T INF = std::numeric_limits<T>::max() / 2 - 5;
    int n;
    vector<pair<T, T>> tree;
 
    LiChao(int _n) : n(max(_n, 1)) {
        int nn = (1 << __lg(n)) * 4;
        tree.assign(nn, {0, -INF});
    }
    
    void add(T k, T b) {
        add_inner({k, b}, 0, 0, n - 1);
    }
 
    void add_inner(pair<T, T> line, int i, int l, int r) {
        while (l <= r) {
            T curl = tree[i].first * l + tree[i].second;
            T curr = tree[i].first * r + tree[i].second;
            T mel = line.first * l + line.second;
            T mer = line.first * r + line.second;
            if (curl >= mel && curr >= mer) {
                break;
            }
            if (curl <= mel && curr <= mer) {
                tree[i] = line;
                break;
            }
            assert(line.first != tree[i].first);
            int m = (l + r) / 2;
            T cur = tree[i].first * m + tree[i].second;
            T me = line.first * m + line.second;
            if (me > cur) {
                swap(line, tree[i]);
            }
            if ((me <= cur && mer > curr) || (me > cur && mer < curr)) {
                l = m + 1;
                i = i * 2 + 2;
            } else {
                r = m;
                i = i * 2 + 1;
            }
        }
    }
 
    void add_segment(T k, T b, int l, int r) {
        l = max(l, 0);
        r = min(r, n - 1);
        if (l > r) return;
        show(k, b, l, r);
        add_segment_inner({k, b}, l, r, 0, 0, n - 1);
    }
 
    void add_segment_inner(pair<T, T> line, int l, int r, int i, int vl, int vr) {
        if (l > vr || r < vl) {
            return;
        }
        if (l <= vl && vr <= r) {
            add_inner(line, i, vl, vr);
            return;
        }
        int m = (vl + vr) / 2;
        add_segment_inner(line, l, r, i * 2 + 1, vl, m);
        add_segment_inner(line, l, r, i * 2 + 2, m + 1, vr);
    }
 
    T ask(int x) {
        T ans = -INF;
        int i = 0;
        int l = 0;
        int r = n - 1;
        while (true) {
            int m = (l + r) / 2;
            ans = max(ans, tree[i].first * x + tree[i].second);
            if (l == r) break;
            if (x <= m) {
                r = m;
                i = i * 2 + 1;
            } else {
                l = m + 1;
                i = i * 2 + 2;
            }
        }
        return ans;
    }
};

struct Line {
    ll k, b;
    ll f(ll x) {
        return k * x + b;
    }
    Line(ll k = 0, ll b = -INF) : k(k), b(b) {}
};

struct Node {
    Line line;
    int left;
    int right;
    Node(Line line) : line(line), left(-1), right(-1) {}
    Node() : line(), left(-1), right(-1) {}
};

struct li_chao_tree {
    int idx;
    vector<Node> nodes;
    int L, R; 

    li_chao_tree(int n, ll L = -inf, ll R = inf) : idx(0), L(L), R(R) {
        nodes.rsz(n);
        nodes[0] = Node(Line());
        idx = 1;
    }

    void add_line(int l, int r, int node, Line cur) {
        if (l > r) return;
        int mid = (l + r) / 2;
        if (r - l == 1 && mid == r) {
            mid--;
        }
        bool lf = cur.f(l) > nodes[node].line.f(l);
        bool md = cur.f(mid) > nodes[node].line.f(mid);
        if (md)
            swap(nodes[node].line, cur);
        if (l == r)
            return;
        if (lf != md) {
            if (nodes[node].left == -1) {
                nodes[node].left = idx;
                nodes[idx++] = Node(cur);
            } else {
                add_line(l, mid, nodes[node].left, cur);
            }
        } else {
            if (nodes[node].right == -1) {
                nodes[node].right = idx;
                nodes[idx++] = Node(cur);
            } else {
                add_line(mid + 1, r, nodes[node].right, cur);
            }
        }
    }

    void add_line(Line new_line) {
        add_line(L, R, 0, new_line);
    }

    ll query(int l, int r, int node, ll x) {
        if (l > r)
            return -INF;
        int mid = (l + r) / 2;
        if (r - l == 1 && mid == r) {
            mid--;
        }
        ll ans = nodes[node].line.f(x);
        if (l == r)
            return ans;
        if (x <= mid && nodes[node].left != -1) {
            ans = max(ans, query(l, mid, nodes[node].left, x));
        }
        if (x > mid && nodes[node].right != -1) {
            ans = max(ans, query(mid + 1, r, nodes[node].right, x));
        }
        return ans;
    }

    ll query(ll x) {
        return query(L, R, 0, x);
    }
};
