# dashboard.py - الجزء الأول
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import os
import traceback

# --- **** يجب أن يكون أول أمر Streamlit **** ---
st.set_page_config(layout="wide", page_title="لوحة تحليل المبيعات والتنبؤ")
# -----------------------------------------------------------------------

# --- استيراد الدوال الجديدة من الملفات الأخرى ---
try:
    from feature_engineering import generate_features_df
    # استيراد دالة التنبؤ واسم عمود التاريخ الناتج منها
    from forecasting import train_and_forecast, DATE_COLUMN_OUTPUT
except ImportError as import_err:
    st.error(f"خطأ في استيراد الدوال: {import_err}")
    st.error("تأكد من وجود ملفي `feature_engineering.py` و `forecasting.py` في نفس مجلد هذا السكربت.")
    st.stop()

# --- إعدادات مسارات البيانات الأصلية ---
SALE_INVOICES_PATH = "data/sale_invoices.xlsx"
SALE_INVOICES_DETAILS_PATH = "data/sale_invoices_details.xlsx"
PRODUCTS_PATH = "data/products.xlsx"
INVOICE_DEFERRED_PATH = "data/invoice_deferreds.xlsx"

# --- دالة مساعدة لتحليل التواريخ ---
def parse_excel_date(date_val):
    """يحاول تحليل التاريخ من الأرقام أو النصوص."""
    if pd.isna(date_val): return pd.NaT
    try:
        if isinstance(date_val, (int, float)):
            if date_val > 59:
                 if date_val > 60: date_val -= 1
                 return pd.Timestamp('1899-12-30') + pd.to_timedelta(date_val, unit='D')
            else: return pd.NaT
        elif isinstance(date_val, (datetime, pd.Timestamp)):
            return pd.to_datetime(date_val)
        elif isinstance(date_val, str):
            common_formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
            for fmt in common_formats:
                try: return pd.to_datetime(date_val, format=fmt)
                except (ValueError, TypeError): continue
            return pd.to_datetime(date_val, errors='coerce')
        else:
            return pd.to_datetime(date_val, errors='coerce')
    except Exception:
        return pd.NaT

class DataPipeline:
    def __init__(self):
        self.raw_data = {}
        self.processed_data = {}
        self.analytics = {
            'daily_sales_actual': pd.DataFrame(),
            'product_flow': pd.DataFrame(),
            'outstanding_amounts': pd.DataFrame(),
            'pareto_data': pd.DataFrame(),
            'pie_data': {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}},
            'stagnant_products': pd.DataFrame(),
        }
        self.visualizations = {}
        self.future_forecast_data = pd.DataFrame()
        self.original_data_paths = {
            'products': PRODUCTS_PATH,
            'sale_invoices': SALE_INVOICES_PATH,
            'sale_invoices_details': SALE_INVOICES_DETAILS_PATH,
            'invoice_deferred': INVOICE_DEFERRED_PATH
        }
        self.feature_data_paths = {
            'sale_invoices': SALE_INVOICES_PATH,
            'sale_invoices_details': SALE_INVOICES_DETAILS_PATH
        }
        self.forecast_horizon = 14

    # --- مرحلة 1: تحميل البيانات الأصلية ---
    @st.cache_data(show_spinner="جاري تحميل البيانات الأصلية (Excel)...")
    def load_original_data(_self):
        """تحميل البيانات الأولية من ملفات Excel."""
        raw_data_loaded = {}
        all_files_found = True
        for name, path in _self.original_data_paths.items():
             if not os.path.exists(path):
                 st.error(f"لم يتم العثور على ملف '{name}': {path}")
                 all_files_found = False
                 raw_data_loaded[name] = pd.DataFrame()
             else:
                 try:
                     dtype_map = {}
                     if name == 'sale_invoices_details': dtype_map = {'quantity':'float32', 'totalPrice':'float32', 'buyPrice':'float32', 'price':'float32', 'discountPrice':'float32', 'invoice_id':'Int64', 'product_id':'Int64'}
                     elif name == 'sale_invoices': dtype_map = {'id':'Int64', 'totalPrice':'float32', 'paidAmount':'float32', 'remainingAmount':'float32', 'user_id':'Int64'}
                     elif name == 'products': dtype_map = {'id':'Int64', 'quantity':'float32', 'buyPrice':'float32', 'salePrice':'float32'}
                     raw_data_loaded[name] = pd.read_excel(path, dtype=dtype_map)
                 except Exception as e:
                     st.error(f"خطأ تحميل {name}: {e}")
                     raw_data_loaded[name] = pd.DataFrame()
                     all_files_found = False

        required_for_analysis = ['sale_invoices', 'sale_invoices_details']
        missing_essentials = [name for name in required_for_analysis if name not in raw_data_loaded or raw_data_loaded[name].empty]
        if missing_essentials:
             st.error(f"الملفات الأساسية مفقودة: {', '.join(missing_essentials)}. لا يمكن المتابعة.")
             st.stop()
        return raw_data_loaded

    # --- مرحلة 2: معالجة البيانات الأصلية وحساب التحليلات ---
    @st.cache_data(show_spinner="جاري معالجة البيانات الأصلية وتحليلها...")
    def preprocess_and_analyze(_self, raw_data):
        """تنظيف، تحضير، وتحليل البيانات الأصلية لحساب كل التحليلات المطلوبة."""
        processed_data = {}
        analytics = {
            'daily_sales_actual': pd.DataFrame(columns=['date', 'actual']),
            'product_flow': pd.DataFrame(), 'outstanding_amounts': pd.DataFrame(),
            'pareto_data': pd.DataFrame(), 'stagnant_products': pd.DataFrame(),
            'pie_data': {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}}
        }

        si_raw = raw_data.get('sale_invoices', pd.DataFrame())
        sid_raw = raw_data.get('sale_invoices_details', pd.DataFrame())
        products_raw = raw_data.get('products', pd.DataFrame())
        deferred_raw = raw_data.get('invoice_deferred', pd.DataFrame())

        if si_raw.empty or sid_raw.empty:
            st.error("بيانات الفواتير أو التفاصيل فارغة.")
            return processed_data, analytics

        si = si_raw.copy()
        sid = sid_raw.copy()
        products = products_raw.copy()
        deferred = deferred_raw.copy()

        # تحديد الأعمدة بمرونة
        si_date_col = next((col for col in ['created_at', 'date', 'invoice_date', 'تاريخ_الإنشاء', 'إنشئ في'] if col in si.columns), None)
        si_id_col = next((col for col in ['id', 'invoice_id', 'رقم_الفاتورة'] if col in si.columns), None)
        si_total_col = next((col for col in ['totalPrice', 'total_price', 'الإجمالي'] if col in si.columns), 'totalPrice')

        sid_invoice_id_col = next((col for col in ['invoice_id', 'InvoiceId', 'رقم_الفاتورة'] if col in sid.columns), None)
        sid_amount_col = next((col for col in ['totalPrice', 'total_price', 'amount', 'الإجمالي'] if col in sid.columns), None)
        sid_quantity_col = next((col for col in ['quantity', 'qty', 'الكمية'] if col in sid.columns), None)
        sid_buy_price_col = next((col for col in ['buyPrice', 'cost_price', 'سعر_الشراء'] if col in sid.columns), None)
        sid_product_id_col = next((col for col in ['product_id', 'item_id', 'معرف_المنتج'] if col in sid.columns), None)
        sid_created_at_col = next((col for col in ['created_at', 'إنشئ_في', 'التاريخ'] if col in sid.columns), None)

        prod_id_col = next((col for col in ['id', 'product_id'] if col in products.columns), None)
        prod_name_col = next((col for col in ['name', 'product_name'] if col in products.columns), None)
        prod_quantity_col = next((col for col in ['quantity', 'stock', 'المخزون'] if col in products.columns), None)
        prod_buy_price_col = next((col for col in ['buyPrice', 'cost'] if col in products.columns), None)
        prod_sale_price_col = next((col for col in ['salePrice', 'price'] if col in products.columns), None)

        # التحقق من الأعمدة الأساسية
        required_missing = []
        if not si_date_col: required_missing.append("عمود التاريخ (si)")
        if not si_id_col: required_missing.append("عمود ID (si)")
        if not sid_invoice_id_col: required_missing.append("عمود invoice_id (sid)")
        if not sid_amount_col: required_missing.append("عمود الإجمالي (sid)")
        if not sid_quantity_col: required_missing.append("عمود الكمية (sid)")
        if not sid_product_id_col: required_missing.append("عمود product_id (sid)")
        if products_raw.empty: required_missing.append("ملف المنتجات فارغ")
        else:
            if not prod_id_col: required_missing.append("عمود ID (products)")
            if not prod_name_col: required_missing.append("عمود الاسم (products)")

        if required_missing:
             st.error("الأعمدة/الملفات الأساسية للتحليل مفقودة: " + ", ".join(required_missing))
             return processed_data, analytics

        # المعالجة
        try:
            si['parsed_date'] = si[si_date_col].apply(parse_excel_date)
            si.dropna(subset=['parsed_date'], inplace=True)
            si.rename(columns={si_id_col: 'invoice_pk'}, inplace=True)
            si['invoice_pk'] = pd.to_numeric(si['invoice_pk'], errors='coerce').astype('Int64')
            si.dropna(subset=['invoice_pk'], inplace=True)
            si_total_col_original = si_total_col
            if si_total_col_original not in si.columns: si_total_col_original = 'totalPrice'
            si[si_total_col_original] = pd.to_numeric(si[si_total_col_original], errors='coerce').fillna(0).astype('float32')
            si['created_at_date'] = pd.to_datetime(si['parsed_date'])
            si['created_at_per_day'] = si['created_at_date'].dt.date
            si['year'] = si['created_at_date'].dt.year.astype('Int16')
            si['month'] = si['created_at_date'].dt.month.astype('Int8')
            si['month_name'] = si['created_at_date'].dt.strftime("%B")
            si['year_month'] = si['created_at_date'].dt.strftime("%Y-%m")

            sid.rename(columns={sid_invoice_id_col: 'invoice_fk', sid_product_id_col: 'product_id'}, inplace=True)
            sid['invoice_fk'] = pd.to_numeric(sid['invoice_fk'], errors='coerce').astype('Int64')
            sid['product_id'] = pd.to_numeric(sid['product_id'], errors='coerce').astype('Int64')
            sid_amount_col_original = sid_amount_col
            if sid_amount_col_original not in sid.columns: sid_amount_col_original = 'totalPrice'
            sid[sid_amount_col_original] = pd.to_numeric(sid[sid_amount_col_original], errors='coerce').fillna(0).astype('float32')
            sid_quantity_col_original = sid_quantity_col
            if sid_quantity_col_original not in sid.columns: sid_quantity_col_original = 'quantity'
            sid[sid_quantity_col_original] = pd.to_numeric(sid[sid_quantity_col_original], errors='coerce').fillna(0).astype('float32')
            sid_buy_price_col_original = sid_buy_price_col
            if sid_buy_price_col_original and sid_buy_price_col_original in sid.columns:
                sid[sid_buy_price_col_original] = pd.to_numeric(sid[sid_buy_price_col_original], errors='coerce').fillna(0).astype('float32')
            else:
                 sid['buyPrice'] = 0.0
                 sid_buy_price_col_original = 'buyPrice'
            sid_created_at_col_original = sid_created_at_col
            if sid_created_at_col_original and sid_created_at_col_original in sid.columns:
                 sid['created_at_dt'] = sid[sid_created_at_col_original].apply(parse_excel_date)
                 sid['created_at_dt'] = pd.to_datetime(sid['created_at_dt'])
            else:
                 sid = pd.merge(sid, si[['invoice_pk', 'created_at_date']], left_on='invoice_fk', right_on='invoice_pk', how='left')
                 sid.rename(columns={'created_at_date':'created_at_dt'}, inplace=True)

            sid['created_at_dt'] = pd.to_datetime(sid['created_at_dt'], errors='coerce')
            sid.dropna(subset=['invoice_fk', 'product_id'], inplace=True)
            sid['netProfit'] = sid[sid_amount_col_original] - (sid[sid_buy_price_col_original] * sid[sid_quantity_col_original])

            products.rename(columns={prod_id_col: 'product_pk', prod_name_col: 'product_name'}, inplace=True)
            products['product_pk'] = pd.to_numeric(products['product_pk'], errors='coerce').astype('Int64')
            prod_quantity_col_original = prod_quantity_col
            if prod_quantity_col_original and prod_quantity_col_original in products.columns:
                products.rename(columns={prod_quantity_col_original: 'quantity'}, inplace=True)
                products['quantity'] = pd.to_numeric(products['quantity'], errors='coerce').fillna(0).astype('float32')
            else: products['quantity'] = 0.0
            prod_buy_price_col_p = prod_buy_price_col
            if prod_buy_price_col_p and prod_buy_price_col_p in products.columns:
                products.rename(columns={prod_buy_price_col_p: 'buyPrice'}, inplace=True)
                products['buyPrice'] = pd.to_numeric(products['buyPrice'], errors='coerce').fillna(0).astype('float32')
            else: products['buyPrice'] = 0.0
            prod_sale_price_col_p = prod_sale_price_col
            if prod_sale_price_col_p and prod_sale_price_col_p in products.columns:
                products.rename(columns={prod_sale_price_col_p: 'salePrice'}, inplace=True)
                products['salePrice'] = pd.to_numeric(products['salePrice'], errors='coerce').fillna(0).astype('float32')
            else: products['salePrice'] = 0.0
            products.dropna(subset=['product_pk', 'product_name'], inplace=True)
            products['name'] = products['product_name']
            products['id'] = products['product_pk']

        except Exception as e_process:
             st.error(f"خطأ أثناء معالجة البيانات الأصلية: {e_process}")
             st.exception(e_process)
             return processed_data, analytics

        # --- حساب التحليلات ---
        try:
            # 1. المبيعات اليومية الفعلية (لـ fig11)
            merged_sales_fig11 = pd.merge(
                sid[['invoice_fk', sid_amount_col_original]],
                si[['invoice_pk', 'created_at_per_day']],
                left_on='invoice_fk', right_on='invoice_pk', how='left'
            )
            merged_sales_fig11.dropna(subset=['created_at_per_day'], inplace=True)
            if not merged_sales_fig11.empty:
                daily_sales_actual_df = merged_sales_fig11.groupby('created_at_per_day')[sid_amount_col_original].sum().reset_index()
                daily_sales_actual_df.rename(columns={'created_at_per_day': 'date', sid_amount_col_original: 'actual'}, inplace=True)
                daily_sales_actual_df['date'] = pd.to_datetime(daily_sales_actual_df['date']).dt.date
                analytics['daily_sales_actual'] = daily_sales_actual_df.sort_values('date')
            else:
                st.warning("لا توجد بيانات لحساب المبيعات اليومية الفعلية.")
                analytics['daily_sales_actual'] = pd.DataFrame(columns=['date', 'actual'])

            # 2. تدفق المنتجات (Product Flow)
            if not products.empty:
                sales_qty_agg = sid.groupby('product_id')[sid_quantity_col_original].sum().reset_index()
                sales_total_agg = sid.groupby('product_id')[sid_amount_col_original].sum().reset_index()
                product_flow_df = pd.merge(
                    sales_qty_agg.rename(columns={sid_quantity_col_original: 'sales_quantity'}),
                    sales_total_agg.rename(columns={sid_amount_col_original: 'sales_amount'}),
                    on='product_id', how='outer').fillna(0)
                product_flow_df = pd.merge(
                    product_flow_df,
                    products[['id', 'name', 'buyPrice', 'salePrice', 'quantity']].rename(columns={'quantity': 'current_stock'}),
                    left_on='product_id', right_on='id', how='left').fillna({'name':'Unknown', 'buyPrice':0, 'salePrice':0, 'current_stock':0})

                product_flow_df['efficiency_ratio'] = product_flow_df['current_stock'] / product_flow_df['sales_quantity'].replace(0, np.nan)
                product_flow_df['efficiency'] = pd.cut(
                    product_flow_df['efficiency_ratio'], bins=[0, 0.8, 1.2, float('inf')],
                    labels=['Undersupplied', 'Balanced', 'Oversupplied'], right=False)
                product_flow_df['efficiency'] = product_flow_df['efficiency'].cat.add_categories('No Sales/Stock').fillna('No Sales/Stock')

                # --- الكود المصحح لحساب days_since_last_sale ---
                if 'created_at_dt' in sid.columns and not sid['created_at_dt'].isnull().all():
                     last_sale = sid.groupby('product_id')['created_at_dt'].max().reset_index()
                     product_flow_df = pd.merge(product_flow_df, last_sale, on='product_id', how='left')
                     product_flow_df.rename(columns={'created_at_dt': 'last_sale_date'}, inplace=True)
                     product_flow_df['last_sale_date'] = pd.to_datetime(product_flow_df['last_sale_date'], errors='coerce')

                     today_date = datetime.now().date()
                     today_ts = pd.Timestamp(today_date)
                     mask_valid_date = product_flow_df['last_sale_date'].notna()
                     product_flow_df.loc[mask_valid_date, 'days_since_last_sale'] = \
                         (today_ts - product_flow_df.loc[mask_valid_date, 'last_sale_date'].dt.normalize()).dt.days
                     product_flow_df['days_since_last_sale'] = pd.to_numeric(product_flow_df['days_since_last_sale'], errors='coerce').fillna(9999).astype(int) # Ensure numeric and fillna
                else:
                     product_flow_df['last_sale_date'] = pd.NaT
                     product_flow_df['days_since_last_sale'] = 9999
                # --- نهاية الكود المصحح ---

                product_profit = sid.groupby('product_id')['netProfit'].sum().reset_index()
                product_flow_df = pd.merge(product_flow_df, product_profit, on='product_id', how='left').fillna({'netProfit': 0})
                product_flow_df['product_name'] = product_flow_df['name']
                analytics['product_flow'] = product_flow_df
            else:
                 st.warning("بيانات المنتجات فارغة، لا يمكن حساب تدفق المنتجات.")
                 analytics['product_flow'] = pd.DataFrame()

            # 3. تحليل باريتو
            if not analytics['product_flow'].empty:
                sorted_products = analytics['product_flow'].sort_values('sales_quantity', ascending=False)
                total_sales_quantity = sorted_products['sales_quantity'].sum()
                if total_sales_quantity > 0:
                    sorted_products['cumulative_percentage'] = sorted_products['sales_quantity'].cumsum() / total_sales_quantity * 100
                else: sorted_products['cumulative_percentage'] = 0
                sorted_products['category'] = (sorted_products['cumulative_percentage'].fillna(0) // 10) * 10
                analytics['pareto_data'] = sorted_products[sorted_products['cumulative_percentage'] <= 80].copy()
            else: analytics['pareto_data'] = pd.DataFrame()

            # 4. تحليل دوائر الإيراد والربح
            if not analytics['product_flow'].empty:
                product_flow_for_pie = analytics['product_flow']
                top_revenue = product_flow_for_pie.sort_values('sales_amount', ascending=False).head(10)
                top_profit = product_flow_for_pie.sort_values('netProfit', ascending=False).head(10)
                total_revenue = product_flow_for_pie['sales_amount'].sum()
                total_profit = product_flow_for_pie['netProfit'].sum()

                top_revenue_pie = top_revenue[['name', 'sales_amount']].rename(columns={'sales_amount':'totalPrice'})
                top_profit_pie = top_profit[['name', 'netProfit']]

                remaining_revenue_val = max(0, total_revenue - top_revenue_pie['totalPrice'].sum())
                remaining_profit_val = max(0, total_profit - top_profit_pie['netProfit'].sum())
                remaining_revenue = pd.DataFrame([{'name': 'Other Products', 'totalPrice': remaining_revenue_val}])
                remaining_profit = pd.DataFrame([{'name': 'Other Products', 'netProfit': remaining_profit_val}])

                pie_revenue_df = pd.concat([top_revenue_pie, remaining_revenue], ignore_index=True) if not top_revenue_pie.empty else remaining_revenue
                pie_profit_df = pd.concat([top_profit_pie, remaining_profit], ignore_index=True) if not top_profit_pie.empty else remaining_profit

                color_map = {row['name']: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, row in top_revenue.iterrows()}
                color_map['Other Products'] = 'gray'

                analytics['pie_data'] = {
                    'revenue': pie_revenue_df[pie_revenue_df['totalPrice'] > 0].copy(),
                    'profit': pie_profit_df[pie_profit_df['netProfit'] > 0].copy(),
                     'color_mapping': color_map
                }
            else: analytics['pie_data'] = {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}}

            # 5. المنتجات الراكدة
            if not analytics['product_flow'].empty and 'days_since_last_sale' in analytics['product_flow'].columns:
                 days_col = pd.to_numeric(analytics['product_flow']['days_since_last_sale'], errors='coerce')
                 stagnant = analytics['product_flow'][
                     (days_col >= 90) & (days_col != 9999)
                 ].copy()
                 if not stagnant.empty:
                     stagnant['days_since_last_sale_num'] = pd.to_numeric(stagnant['days_since_last_sale'], errors='coerce')
                     stagnant.dropna(subset=['days_since_last_sale_num'], inplace=True)

                     stagnant['days_category'] = pd.cut(
                         stagnant['days_since_last_sale_num'], bins=[90, 270, 365, float('inf')],
                         labels=['3-9 أشهر', '9-12 شهر', 'أكثر من سنة'], right=False)
                     analytics['stagnant_products'] = stagnant.sort_values('days_since_last_sale_num', ascending=False).drop(columns=['days_since_last_sale_num'])
                 else: st.info("لا توجد منتجات راكدة (لم تباع منذ 90 يومًا أو أكثر).")
            else: analytics['stagnant_products'] = pd.DataFrame()

            # 6. المبالغ الآجلة
            if not deferred.empty:
                def_type_col = 'invoice_type'; def_status_col = 'status'
                def_amount_col = 'amount'; def_paid_col = 'paid_amount'; def_user_id_col = 'user_id'
                required_deferred_cols = [def_type_col, def_status_col, def_amount_col, def_paid_col, def_user_id_col]

                if all(col in deferred.columns for col in required_deferred_cols):
                    deferred[def_amount_col] = pd.to_numeric(deferred[def_amount_col], errors='coerce').fillna(0)
                    deferred[def_paid_col] = pd.to_numeric(deferred[def_paid_col], errors='coerce').fillna(0)
                    buy_invoice_type_str = "Stocks\\Models\\BuyInvoice"
                    status_values = [0, 2]
                    filtered = deferred[
                        (deferred[def_type_col] == buy_invoice_type_str) &
                        (deferred[def_status_col].isin(status_values))
                    ].copy()
                    if not filtered.empty:
                        filtered["outstanding_amount"] = filtered[def_amount_col] - filtered[def_paid_col]
                        outstanding_grouped = filtered.groupby(def_user_id_col, as_index=False)["outstanding_amount"].sum()
                        analytics['outstanding_amounts'] = outstanding_grouped[outstanding_grouped['outstanding_amount'] > 0].copy()
                    else: st.info("لا توجد فواتير شراء آجلة بالمواصفات المحددة.")
                else:
                     missing_cols = [col for col in required_deferred_cols if col not in deferred.columns]
                     st.warning(f"أعمدة مفقودة في بيانات الدفع الآجل: {', '.join(missing_cols)}")
                     analytics['outstanding_amounts'] = pd.DataFrame()
            else:
                st.info("لا توجد بيانات للدفع الآجل.")
                analytics['outstanding_amounts'] = pd.DataFrame()

        except Exception as e_analyze:
             st.error(f"حدث خطأ أثناء حساب التحليلات: {e_analyze}")
             st.exception(e_analyze)

        # تخزين البيانات المعالجة
        processed_data['sale_invoices'] = si
        processed_data['sale_invoices_details'] = sid
        processed_data['products'] = products
        processed_data['invoice_deferred'] = deferred

        return processed_data, analytics

# --- End of Part 1 ---
# dashboard.py - الجزء الثاني (استكمال الجزء الأول)

    # --- مرحلة 3: إنشاء الرسوم البيانية ---
    def generate_visualizations(self):
        """إنشاء جميع الرسوم البيانية المطلوبة."""
        self.visualizations = {} # إعادة تهيئة

        # --- استعادة كود إنشاء الرسوم من 1 إلى 10 و 12 إلى 17 ---
        product_flow = self.analytics.get('product_flow', pd.DataFrame())
        sid_data = self.processed_data.get('sale_invoices_details', pd.DataFrame()) # Get processed data
        si_data = self.processed_data.get('sale_invoices', pd.DataFrame())
        products_data = self.processed_data.get('products', pd.DataFrame())

        # --- fig1 ---
        fig1_title = "📊 مقارنة المخزون والمبيعات (Top 20)"
        if not product_flow.empty:
            top20_qty = product_flow.sort_values('sales_quantity', ascending=False).head(20)
            fig1 = go.Figure()
            if not top20_qty.empty:
                 fig1.add_trace(go.Bar(x=top20_qty['name'], y=top20_qty['current_stock'], name='المخزون الحالي', marker_color='blue'))
                 fig1.add_trace(go.Bar(x=top20_qty['name'], y=top20_qty['sales_quantity'], name='المبيعات', marker_color='orange'))
                 fig1.update_layout(title=fig1_title, xaxis_title="المنتج", yaxis_title="الكمية", barmode="group")
            else: fig1.update_layout(title=fig1_title).add_annotation(text="لا توجد بيانات لعرض Top 20", showarrow=False)
            self.visualizations['fig1'] = fig1
        else: self.visualizations['fig1'] = go.Figure().update_layout(title=fig1_title).add_annotation(text="بيانات تدفق المنتجات مفقودة", showarrow=False)

        # --- fig2 ---
        fig2_title = "📈 توزيع الكفاءة"
        if not product_flow.empty and 'efficiency' in product_flow.columns:
            valid_efficiency_cats = ['Undersupplied', 'Balanced', 'Oversupplied']
            efficiency_counts = product_flow[product_flow['efficiency'].isin(valid_efficiency_cats)]['efficiency'].value_counts()
            if not efficiency_counts.empty:
                 fig2 = px.pie(values=efficiency_counts.values, names=efficiency_counts.index, title=fig2_title,
                               color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
                 self.visualizations['fig2'] = fig2
            else: self.visualizations['fig2'] = go.Figure().update_layout(title=fig2_title).add_annotation(text="لا توجد بيانات كفاءة صالحة", showarrow=False)
        else: self.visualizations['fig2'] = go.Figure().update_layout(title=fig2_title).add_annotation(text="بيانات الكفاءة مفقودة", showarrow=False)

        # --- fig3 ---
        fig3_title = "💰 تحليل الإيرادات (مع COGS)"
        if not product_flow.empty and 'buyPrice' in product_flow.columns and 'sales_quantity' in product_flow.columns and 'name' in product_flow.columns:
            top20_amount = product_flow.sort_values('sales_amount', ascending=False).head(20).copy()
            if not top20_amount.empty:
                top20_amount['COGS'] = top20_amount['sales_quantity'] * top20_amount['buyPrice']
                top20_amount['margin'] = top20_amount['sales_amount'] - top20_amount['COGS']
                fig3 = px.bar(top20_amount, x="name", y=["COGS", "sales_amount", "margin"], barmode="group",
                             title=fig3_title, labels={'value': 'المبلغ', 'variable': 'النوع', 'name':'المنتج'},
                             color_discrete_map={"COGS": "blue", "sales_amount": "orange", "margin": "green"})
                self.visualizations['fig3'] = fig3
            else: self.visualizations['fig3'] = go.Figure().update_layout(title=fig3_title).add_annotation(text="لا توجد بيانات لعرض Top 20", showarrow=False)
        else: self.visualizations['fig3'] = go.Figure().update_layout(title=fig3_title).add_annotation(text="بيانات غير كافية", showarrow=False)

        # --- fig4 ---
        fig4_title = "📦 المخزون الحالي مقابل المبيعات"
        if not product_flow.empty:
            if 'fig1' in self.visualizations and self.visualizations['fig1'].data: # Check if fig1 has data
                 top20_names_fig1 = self.visualizations['fig1'].data[0].x
                 top20_qty_fig4 = product_flow[product_flow['name'].isin(top20_names_fig1)].head(20).sort_values('sales_quantity', ascending=False)
            else:
                 top20_qty_fig4 = product_flow.sort_values('sales_quantity', ascending=False).head(20)

            if not top20_qty_fig4.empty:
                 fig4 = go.Figure()
                 fig4.add_trace(go.Bar(x=top20_qty_fig4['name'], y=top20_qty_fig4['current_stock'], name="المخزون الحالي", marker_color='blue'))
                 fig4.add_trace(go.Scatter(x=top20_qty_fig4['name'], y=top20_qty_fig4['sales_quantity'], mode='lines+markers', name="المبيعات", line=dict(color='orange')))
                 fig4.update_layout(title=fig4_title, xaxis_title="المنتج", yaxis_title="الكمية", barmode="overlay")
                 self.visualizations['fig4'] = fig4
            else: self.visualizations['fig4'] = go.Figure().update_layout(title=fig4_title).add_annotation(text="لا توجد بيانات لعرض Top 20", showarrow=False)
        else: self.visualizations['fig4'] = go.Figure().update_layout(title=fig4_title).add_annotation(text="بيانات تدفق المنتجات مفقودة", showarrow=False)

        # --- fig5 ---
        fig5_title = "⚖️ مقارنة الكفاءة"
        if not product_flow.empty and 'efficiency' in product_flow.columns:
             valid_cats = ['Undersupplied', 'Balanced', 'Oversupplied']
             eff_group = product_flow[product_flow['efficiency'].isin(valid_cats)].groupby('efficiency', observed=False)[['current_stock', 'sales_quantity']].sum().reset_index()
             if not eff_group.empty:
                  fig5 = go.Figure()
                  fig5.add_trace(go.Bar(x=eff_group['efficiency'], y=eff_group['current_stock'], name='المخزون الحالي', marker_color='blue', opacity=0.6))
                  fig5.add_trace(go.Bar(x=eff_group['efficiency'], y=eff_group['sales_quantity'], name='المبيعات', marker_color='orange', opacity=0.6))
                  fig5.update_layout(barmode='group', title=fig5_title, xaxis_title="الفئة", yaxis_title="الكمية")
                  self.visualizations['fig5'] = fig5
             else: self.visualizations['fig5'] = go.Figure().update_layout(title=fig5_title).add_annotation(text="لا توجد بيانات كفاءة صالحة", showarrow=False)
        else: self.visualizations['fig5'] = go.Figure().update_layout(title=fig5_title).add_annotation(text="بيانات الكفاءة مفقودة", showarrow=False)

        # --- fig6 ---
        # --- fig6 ---
        fig6_title = "📊 تحليل باريتو (أعلى 80% من المبيعات)"
        # --- استخدم البيانات المصفاة ---
        pareto_data = self.analytics.get('pareto_data', pd.DataFrame()) # <-- البيانات المصفاة (حتى 80%)

        if not pareto_data.empty and all(col in pareto_data.columns for col in ['name', 'sales_quantity', 'cumulative_percentage', 'category']):

            fig6 = go.Figure()

            # 1. رسم الأعمدة (للمنتجات <= 80%)
            # *** ملاحظة: الترتيب هنا قد لا يكون حسب المبيعات بالضرورة إذا لم نفرزه ***
            # لكن لنفرزه ليكون المحور السيني أكثر منطقية
            pareto_data_sorted_sales = pareto_data.sort_values('sales_quantity', ascending=False)
            fig6.add_trace(go.Bar(
                x=pareto_data_sorted_sales['name'],
                y=pareto_data_sorted_sales['sales_quantity'],
                name="المبيعات (أعلى 80%)",
                marker_color='blue'
            ))

            # 2. رسم الخط التراكمي المقسم والملون (مثل الكود الثاني)
            colors = px.colors.qualitative.Plotly
            # *** نستخدم pareto_data مباشرة في الحلقة، لا حاجة لفرز منفصل حسب الفئة هنا ***
            valid_categories = sorted([cat for cat in pareto_data['category'].unique() if isinstance(cat, (int, float)) and pd.notna(cat)])

            for i, category in enumerate(valid_categories):
                # *** جلب البيانات من pareto_data مباشرة ***
                cat_data = pareto_data[pareto_data['category'] == category]
                # *** لا يوجد reindex ***
                if not cat_data.empty:
                    fig6.add_trace(go.Scatter(
                        x=cat_data['name'],          # <-- من cat_data
                        y=cat_data['cumulative_percentage'], # <-- من cat_data
                        name=f"{int(category)}-{int(category + 10)}%",
                        mode='lines+markers',
                        yaxis="y2",
                        line=dict(color=colors[i % len(colors)], dash='dash') # خط منقط ملون
                    ))

            # 3. تحديث تخطيط الرسمة (بدون categoryorder)
            fig6.update_layout(
                title=fig6_title,
                xaxis_title="المنتج", # اسم عام للمحور السيني
                yaxis_title="المبيعات",
                yaxis2=dict(
                    title="النسبة التراكمية (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 85] # المدى مناسب للـ 80%
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                # *** تم حذف xaxis={'categoryorder':...} ***
            )
            self.visualizations['fig6'] = fig6
        else:
            self.visualizations['fig6'] = go.Figure().update_layout(title=fig6_title).add_annotation(text="بيانات باريتو غير كافية", showarrow=False)         
        # --- fig7 ---
        fig7_title = "📈 تحليل تسعير المنتجات"
        if not product_flow.empty and 'salePrice' in product_flow.columns and 'sales_quantity' in product_flow.columns:
             fig7_data = product_flow[(product_flow['salePrice'] > 0) & (product_flow['sales_quantity'] > 0)]
             if not fig7_data.empty:
                 fig7 = px.scatter(fig7_data, x="salePrice", y="sales_quantity", title=fig7_title,
                                  labels={'salePrice': 'سعر البيع', 'sales_quantity': 'حجم المبيعات'},
                                  trendline="lowess", hover_data=['name'])
                 self.visualizations['fig7'] = fig7
             else: self.visualizations['fig7'] = go.Figure().update_layout(title=fig7_title).add_annotation(text="لا توجد بيانات سعر مقابل طلب", showarrow=False)
        else: self.visualizations['fig7'] = go.Figure().update_layout(title=fig7_title).add_annotation(text="بيانات السعر/الكمية مفقودة", showarrow=False)

        # --- fig9 ---
        fig9_title = "📦 منتجات تحتاج لإعادة تخزين (<10)"
        if not product_flow.empty and 'current_stock' in product_flow.columns and 'name' in product_flow.columns:
            restock = product_flow[product_flow['current_stock'] <= 10].sort_values('current_stock')
            if not restock.empty:
                 fig9 = px.bar(restock, x="name", y="current_stock", title=fig9_title,
                              labels={'name': 'المنتج', 'current_stock': 'المخزون الحالي'})
                 fig9.update_layout(xaxis={'categoryorder':'total ascending'})
                 self.visualizations['fig9'] = fig9
            else: self.visualizations['fig9'] = go.Figure().update_layout(title=fig9_title).add_annotation(text="لا توجد منتجات تحتاج إعادة تخزين", showarrow=False)
        else: self.visualizations['fig9'] = go.Figure().update_layout(title=fig9_title).add_annotation(text="بيانات المخزون مفقودة", showarrow=False)

        # --- fig10 ---
        fig10_title = '📅 المنتجات الراكدة (90+ يوم)'
        stagnant_products_orig = self.analytics.get('stagnant_products', pd.DataFrame())
        if not stagnant_products_orig.empty:
             stagnant_products_for_plot = stagnant_products_orig.copy()
             hover_data_dict_str = {}
             if 'product_id' in stagnant_products_for_plot.columns:
                 stagnant_products_for_plot['product_id_str'] = stagnant_products_for_plot['product_id'].astype(str).fillna('N/A')
                 hover_data_dict_str['ID'] = stagnant_products_for_plot['product_id_str']
             if 'last_sale_date' in stagnant_products_for_plot.columns:
                 stagnant_products_for_plot['last_sale_date'] = pd.to_datetime(stagnant_products_for_plot['last_sale_date'], errors='coerce')
                 stagnant_products_for_plot['last_sale_date_str'] = stagnant_products_for_plot['last_sale_date'].dt.strftime('%Y-%m-%d').fillna('N/A')
                 hover_data_dict_str['آخر بيع'] = stagnant_products_for_plot['last_sale_date_str']
             if 'current_stock' in stagnant_products_for_plot.columns:
                 stagnant_products_for_plot['current_stock_str'] = stagnant_products_for_plot['current_stock'].round(1).astype(str).fillna('N/A')
                 hover_data_dict_str['المخزون'] = stagnant_products_for_plot['current_stock_str']

             required_cols_fig10 = ['name', 'days_since_last_sale', 'days_category']
             if all(c in stagnant_products_for_plot.columns for c in required_cols_fig10):
                 try:
                     stagnant_products_for_plot['days_since_last_sale'] = pd.to_numeric(stagnant_products_for_plot['days_since_last_sale'], errors='coerce')
                     stagnant_products_for_plot.dropna(subset=['days_since_last_sale'], inplace=True)

                     fig10 = px.bar(stagnant_products_for_plot, x='name', y='days_since_last_sale', color='days_category',
                                    title=fig10_title,
                                    labels={'name': 'المنتج', 'days_since_last_sale': 'أيام منذ آخر بيع', 'days_category': 'فترة الركود'},
                                    color_discrete_map={'3-9 أشهر': '#0000FF', '9-12 شهر': '#FF6347', 'أكثر من سنة': '#DC143C'},
                                    hover_data=hover_data_dict_str)
                     fig10.update_layout(xaxis_title="المنتج", yaxis_title="أيام منذ آخر بيع", xaxis={'categoryorder':'total descending'}, hovermode="x unified")
                     fig10.add_hline(y=180, line_dash="dot", annotation_text="حد 6 أشهر", annotation_position="top right", line_color="red")
                     self.visualizations['fig10'] = fig10
                 except Exception as e_fig10:
                      st.error(f"خطأ إنشاء الرسم 10: {e_fig10}")
                      self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="خطأ إنشاء", showarrow=False)
             else: self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="بيانات غير كافية", showarrow=False)
        else: self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="لا توجد منتجات راكدة", showarrow=False)


        # --- fig11: الكود الجديد والمحدث ---
        # --- fig11: الكود الجديد والمحدث ---
        fig11_title = "📈 المبيعات اليومية: الفعلية والمتوقعة"
        daily_sales_actual_df_fig11 = self.analytics.get('daily_sales_actual', pd.DataFrame())
        future_forecast_df_fig11 = self.future_forecast_data
        fig11 = go.Figure()
        last_actual_point = None

        if isinstance(daily_sales_actual_df_fig11, pd.DataFrame) and not daily_sales_actual_df_fig11.empty:
             try:
                 df_actual_plot = daily_sales_actual_df_fig11.copy()
                 df_actual_plot['date'] = pd.to_datetime(df_actual_plot['date'])
                 df_actual_plot = df_actual_plot.sort_values('date')
                 if not df_actual_plot.empty:
                      last_actual_point = df_actual_plot.iloc[-1]

                 df_actual_plot['date_display'] = df_actual_plot['date'].dt.date

                 fig11.add_trace(go.Scatter(
                     x=df_actual_plot['date_display'],
                     y=df_actual_plot['actual'],
                     mode='lines+markers',
                     name='المبيعات الفعلية',
                     marker=dict(color='rgba(0, 116, 217, 0.8)', size=5),
                     line=dict(color='rgba(0, 116, 217, 0.8)', width=2)))
             except Exception as e_plot_actual:
                 st.warning(f"خطأ رسم فعلية fig11: {e_plot_actual}")
                 last_actual_point = None

        if isinstance(future_forecast_df_fig11, pd.DataFrame) and not future_forecast_df_fig11.empty and last_actual_point is not None:
            forecast_to_plot = future_forecast_df_fig11.copy()
            if DATE_COLUMN_OUTPUT in forecast_to_plot.columns:
                forecast_to_plot[DATE_COLUMN_OUTPUT] = pd.to_datetime(forecast_to_plot[DATE_COLUMN_OUTPUT])
            else:
                st.warning(f"عمود التاريخ '{DATE_COLUMN_OUTPUT}' مفقود في بيانات التنبؤ.")
                forecast_to_plot = pd.DataFrame()

            if 'forecast' in forecast_to_plot.columns and not forecast_to_plot.empty:
                try:
                    forecast_to_plot = forecast_to_plot.sort_values(DATE_COLUMN_OUTPUT)

                    last_actual_df_row = pd.DataFrame([{
                        DATE_COLUMN_OUTPUT: last_actual_point['date'],
                        'forecast': last_actual_point['actual'],
                        'lower_ci': np.nan if 'lower_ci' in forecast_to_plot.columns else None,
                        'upper_ci': np.nan if 'upper_ci' in forecast_to_plot.columns else None
                    }])
                    cols_to_keep = [col for col in last_actual_df_row.columns if col in forecast_to_plot.columns]
                    last_actual_df_row = last_actual_df_row[cols_to_keep]

                    forecast_to_plot_future_only = forecast_to_plot[forecast_to_plot[DATE_COLUMN_OUTPUT] > last_actual_point['date']].copy()
                    connected_forecast_df = pd.concat([last_actual_df_row, forecast_to_plot_future_only], ignore_index=True)
                    connected_forecast_df['date_display'] = connected_forecast_df[DATE_COLUMN_OUTPUT].dt.date

                    if not connected_forecast_df.empty:
                        fig11.add_trace(go.Scatter(
                            x=connected_forecast_df['date_display'],
                            y=connected_forecast_df['forecast'],
                            mode='lines',
                            name='المبيعات المتوقعة',
                            line=dict(color='rgba(255, 65, 54, 0.9)', dash='dash', width=2)))

                        if 'lower_ci' in connected_forecast_df.columns and 'upper_ci' in connected_forecast_df.columns:
                            ci_valid = connected_forecast_df.iloc[1:].dropna(subset=['lower_ci', 'upper_ci'])
                            if not ci_valid.empty:
                                fig11.add_trace(go.Scatter(
                                    x=ci_valid['date_display'].tolist() + ci_valid['date_display'].tolist()[::-1],
                                    y=ci_valid['upper_ci'].tolist() + ci_valid['lower_ci'].tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255, 65, 54, 0.15)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='فترة الثقة 95%',
                                    showlegend=True,
                                    hoverinfo='skip'
                                ))
                except Exception as e_plot_forecast:
                    st.warning(f"خطأ رسم تنبؤ fig11: {e_plot_forecast}")
            else:
                st.warning(f"DataFrame التنبؤات لـ fig11 يفتقد لعمود 'forecast' أو فارغ.")
        else:
             if last_actual_point is None:
                  st.warning("لم يتم العثور على آخر نقطة فعلية لرسم التنبؤات.")

        fig11.update_layout(
            title=fig11_title,
            xaxis_title="التاريخ",
            yaxis_title="إجمالي المبيعات",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        if last_actual_point is not None:
            try:
                last_actual_dt = last_actual_point['date']
                start_date_plot_dt = last_actual_dt - timedelta(days=90)
                end_date_plot_dt = last_actual_dt + timedelta(days=self.forecast_horizon + 7)

                if isinstance(daily_sales_actual_df_fig11, pd.DataFrame) and not daily_sales_actual_df_fig11.empty:
                     first_actual_dt = pd.to_datetime(daily_sales_actual_df_fig11['date']).min()
                     if pd.notna(first_actual_dt) and start_date_plot_dt < first_actual_dt:
                          start_date_plot_dt = first_actual_dt

                if isinstance(future_forecast_df_fig11, pd.DataFrame) and not future_forecast_df_fig11.empty and DATE_COLUMN_OUTPUT in future_forecast_df_fig11.columns:
                     max_forecast_dt = pd.to_datetime(future_forecast_df_fig11[DATE_COLUMN_OUTPUT]).max()
                     if pd.notna(max_forecast_dt) and max_forecast_dt < end_date_plot_dt:
                          end_date_plot_dt = max_forecast_dt + timedelta(days=1)

                fig11.update_xaxes(range=[start_date_plot_dt, end_date_plot_dt])
            except Exception as e_xaxis:
                st.warning(f"خطأ تحديد نطاق X لـ fig11: {e_xaxis}")

        self.visualizations['fig11'] = fig11
        # --- نهاية fig11 ---

        # --- fig12 ---
        fig12_title = "متوسط الفاتورة الشهرية"
        si_total_col_fig12 = next((col for col in ['totalPrice', 'total_price', 'الإجمالي'] if col in si_data.columns), 'totalPrice')
        if not si_data.empty and all(c in si_data.columns for c in ['month_name', si_total_col_fig12, 'year', 'month']):
            monthly_avg = si_data.groupby(["year","month", "month_name"], as_index=False)[si_total_col_fig12].mean().sort_values(['year', 'month'])
            if not monthly_avg.empty:
                 monthly_avg['display_month'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month_name']
                 fig12 = px.line(monthly_avg, x="display_month", y=si_total_col_fig12, title=fig12_title,
                                labels={"display_month": "الشهر", si_total_col_fig12: "متوسط قيمة الفاتورة"}, markers=True)
                 fig12.update_xaxes(type='category')
                 self.visualizations['fig12'] = fig12
            else: self.visualizations['fig12'] = go.Figure().update_layout(title=fig12_title).add_annotation(text="لا توجد بيانات شهرية", showarrow=False)
        else: self.visualizations['fig12'] = go.Figure().update_layout(title=fig12_title).add_annotation(text="أعمدة مطلوبة مفقودة", showarrow=False)

        # --- fig13 ---
        fig13_title = "أقل 10 منتجات مبيعًا"
        if not product_flow.empty and 'sales_quantity' in product_flow.columns and 'name' in product_flow.columns:
            sold_products = product_flow[product_flow['sales_quantity'] > 0].copy()
            if not sold_products.empty:
                bottom10 = sold_products.sort_values('sales_quantity').head(10)
                fig13 = px.bar(bottom10, x="sales_quantity", y="name", orientation='h', title=fig13_title,
                            labels={"name": "المنتج", "sales_quantity": "الكمية المباعة"},
                            color="sales_quantity", color_continuous_scale=px.colors.sequential.Viridis,
                            hover_data=["product_id"])
                fig13.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title="الكمية المباعة", yaxis_title="المنتج")
                self.visualizations['fig13'] = fig13
            else: self.visualizations['fig13'] = go.Figure().update_layout(title=fig13_title).add_annotation(text="لا توجد منتجات تم بيعها", showarrow=False)
        else: self.visualizations['fig13'] = go.Figure().update_layout(title=fig13_title).add_annotation(text="بيانات تدفق المنتجات مفقودة", showarrow=False)

        # --- fig14 ---
        fig14_title_display = "📊 جدول تكرار كميات البيع"
        sid_qty_col_fig14 = next((col for col in ['quantity', 'qty', 'الكمية'] if col in sid_data.columns), 'quantity')
        if not sid_data.empty and 'product_id' in sid_data.columns and sid_qty_col_fig14 in sid_data.columns:
             sales_per_product = sid_data.groupby("product_id")[sid_qty_col_fig14].sum()
             sales_per_product = sales_per_product[pd.to_numeric(sales_per_product, errors='coerce').notnull() & (sales_per_product > 0)]
             if not sales_per_product.empty:
                 sales_freq = sales_per_product.value_counts().reset_index()
                 if len(sales_freq.columns) == 2:
                      sales_freq.columns = ["الكمية المباعة", "عدد المنتجات"]
                      fig14_data = sales_freq.sort_values(by="الكمية المباعة").head(15)
                      if not fig14_data.empty:
                           fig14_data["الكمية المباعة"] = fig14_data["الكمية المباعة"].round(1).astype(str)
                           fig14_data["عدد المنتجات"] = fig14_data["عدد المنتجات"].astype(str)
                           fig14 = ff.create_table(fig14_data, index=False)
                           self.visualizations['fig14'] = fig14
                           self.visualizations['fig14_title'] = fig14_title_display
                      else: self.visualizations['fig14'] = None; self.visualizations['fig14_title'] = fig14_title_display
                 else: self.visualizations['fig14'] = None; self.visualizations['fig14_title'] = fig14_title_display
             else: self.visualizations['fig14'] = None; self.visualizations['fig14_title'] = fig14_title_display
        else: self.visualizations['fig14'] = None; self.visualizations['fig14_title'] = fig14_title_display


        # --- fig15 ---
        fig15_title = "أكثر 10 منتجات مبيعًا"
        if not product_flow.empty and 'sales_quantity' in product_flow.columns and 'name' in product_flow.columns:
            top10 = product_flow.sort_values('sales_quantity', ascending=False).head(10)
            if not top10.empty and top10['sales_quantity'].iloc[0] > 0:
                 fig15 = px.bar(top10, x="sales_quantity", y="name", orientation='h', title=fig15_title,
                               labels={"name": "المنتج", "sales_quantity": "الكمية المباعة"},
                               color="sales_quantity", color_continuous_scale=px.colors.sequential.Viridis,
                               hover_data=["product_id"])
                 fig15.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title="الكمية المباعة", yaxis_title="المنتج")
                 self.visualizations['fig15'] = fig15
            else: self.visualizations['fig15'] = go.Figure().update_layout(title=fig15_title).add_annotation(text="لا توجد بيانات كافية", showarrow=False)
        else: self.visualizations['fig15'] = go.Figure().update_layout(title=fig15_title).add_annotation(text="بيانات تدفق المنتجات مفقودة", showarrow=False)

        # --- fig16 ---
        fig16_title = "أعلى 10 منتجات ربحًا وإيرادًا"
        pie_data = self.analytics.get('pie_data')
        if pie_data and isinstance(pie_data, dict) and \
           'revenue' in pie_data and not pie_data['revenue'].empty and \
           'profit' in pie_data and not pie_data['profit'].empty and \
           'color_mapping' in pie_data:

            fig16 = make_subplots(rows=1, cols=2, subplot_titles=["الأعلى إيرادًا", "الأعلى ربحًا"],
                                 specs=[[{"type": "domain"}, {"type": "domain"}]])
            revenue_to_plot = pie_data['revenue']
            profit_to_plot = pie_data['profit']

            if not revenue_to_plot.empty:
                fig16.add_trace(go.Pie(labels=revenue_to_plot["name"], values=revenue_to_plot["totalPrice"], name="إيراد",
                                      marker=dict(colors=[pie_data['color_mapping'].get(n, "gray") for n in revenue_to_plot["name"]])), row=1, col=1)
            else: fig16.add_annotation(text="لا يوجد إيراد", xref="paper", yref="paper", x=0.18, y=0.5, showarrow=False)

            if not profit_to_plot.empty:
                fig16.add_trace(go.Pie(labels=profit_to_plot["name"], values=profit_to_plot["netProfit"], name="ربح",
                                      marker=dict(colors=[pie_data['color_mapping'].get(n, "gray") for n in profit_to_plot["name"]])), row=1, col=2)
            else: fig16.add_annotation(text="لا يوجد ربح", xref="paper", yref="paper", x=0.82, y=0.5, showarrow=False)

            fig16.update_traces(hoverinfo='label+percent+value', textinfo='percent', textfont_size=11, insidetextorientation='radial')
            fig16.update_layout(height=600, showlegend=False, title_text=fig16_title)
            self.visualizations['fig16'] = fig16
        else:
            self.visualizations['fig16'] = go.Figure().update_layout(title=fig16_title).add_annotation(text="بيانات الإيراد/الربح غير كافية", showarrow=False)

        # --- fig17 ---
        fig17_title = "الموردين حسب المبلغ الآجل"
        outstanding = self.analytics.get('outstanding_amounts', pd.DataFrame())
        if not outstanding.empty and 'user_id' in outstanding.columns and 'outstanding_amount' in outstanding.columns:
             total_outstanding = outstanding["outstanding_amount"].sum()
             outstanding_sorted = outstanding.sort_values("outstanding_amount", ascending=False)
             outstanding_sorted['user_id'] = outstanding_sorted['user_id'].astype(str)
             fig17 = px.bar(outstanding_sorted, x="user_id", y="outstanding_amount", color="outstanding_amount",
                            title=fig17_title, labels={"user_id": "المورد", "outstanding_amount": "المبلغ الآجل"},
                            color_continuous_scale="Inferno")
             fig17.update_coloraxes(colorbar_title="المبلغ الآجل")
             fig17.update_layout(xaxis={'categoryorder': 'total descending', 'type': 'category'})
             fig17.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98,
                              text=f"مجموع الآجل: {round(total_outstanding,2)}", showarrow=False,
                              font=dict(size=18, color="black"), bgcolor="rgba(211,211,211,0.7)",
                              bordercolor="black", borderwidth=1, borderpad=4, align="right")
             self.visualizations['fig17'] = fig17
        else: self.visualizations['fig17'] = go.Figure().update_layout(title=fig17_title).add_annotation(text="لا توجد بيانات مبالغ آجلة", showarrow=False)


# --- End of Part 2 ---
# dashboard.py - الجزء الثالث (استكمال الجزء الثاني)

    # --- مرحلة 4: تشغيل خط الأنابيب ---
    def run_pipeline(self):
        """تشغيل خطوات تحميل البيانات، هندسة الميزات، التنبؤ، التحليل، وإنشاء الرسوم."""
        pipeline_success = True
        features_df = None

        # 1. تحميل البيانات الأصلية
        try:
            self.raw_data = self.load_original_data()
        except Exception as e_load:
            st.error(f"فشل تحميل البيانات الأصلية: {e_load}")
            st.exception(e_load); pipeline_success = False; st.stop()

        # 2. هندسة الميزات
        if pipeline_success:
             with st.spinner("جاري إنشاء الميزات..."):
                 try:
                     features_df = generate_features_df(self.feature_data_paths)
                     if features_df is None or features_df.empty:
                         st.error("فشلت هندسة الميزات (الناتج فارغ).")
                         pipeline_success = False
                 except Exception as e_feat:
                     st.error(f"فشل في هندسة الميزات: {e_feat}")
                     st.exception(e_feat); pipeline_success = False

        # 3. التدريب والتنبؤ
        if pipeline_success and features_df is not None:
             with st.spinner(f"جاري التدريب والتنبؤ لـ {self.forecast_horizon} يومًا..."):
                 try:
                     self.future_forecast_data = train_and_forecast(features_df=features_df, forecast_horizon=self.forecast_horizon)
                     if self.future_forecast_data is None:
                         st.error("فشلت عملية التدريب والتنبؤ (الناتج None).")
                         pipeline_success = False
                     elif self.future_forecast_data.empty:
                         st.warning("اكتمل التدريب، لكن لم يتم إنشاء تنبؤات (DataFrame فارغ).")
                 except Exception as e_forecast:
                      st.error(f"فشل غير متوقع في التدريب/التنبؤ: {e_forecast}")
                      st.exception(e_forecast); self.future_forecast_data = None; pipeline_success = False

        # 4. معالجة البيانات الأصلية وحساب التحليلات
        if pipeline_success:
             try:
                 processed_data_result, analytics_result = self.preprocess_and_analyze(self.raw_data)
                 self.processed_data = processed_data_result
                 self.analytics = analytics_result # تحديث الحالة بالكائن المرتجع
                 is_analytics_valid = isinstance(self.analytics, dict) and self.analytics.get('daily_sales_actual') is not None and not self.analytics['daily_sales_actual'].empty
                 if not is_analytics_valid:
                      st.warning("لم يتم حساب بيانات التحليل الأساسية (مثل المبيعات الفعلية). قد تكون بعض الرسوم فارغة.")
             except Exception as e_analyze:
                 st.error(f"فشل في معالجة/تحليل البيانات الأصلية: {e_analyze}")
                 st.exception(e_analyze); pipeline_success = False

        # 5. إنشاء الرسوم البيانية (الآن لجميع الرسوم)
        if pipeline_success:
             try:
                 # تأكد من أن analytics يحتوي على البيانات اللازمة قبل استدعاء generate_visualizations
                 if isinstance(self.analytics, dict) and self.analytics:
                     self.generate_visualizations() # يجب أن تنشئ الآن جميع الرسوم
                     if not self.visualizations:
                          st.warning("لم يتم إنشاء أي رسوم بيانية لسبب ما.")
                 else:
                      st.warning("لا يمكن إنشاء الرسوم البيانية بسبب عدم توفر بيانات التحليل.")

             except Exception as e_viz:
                 st.error(f"فشل في إنشاء الرسوم البيانية: {e_viz}")
                 st.exception(e_viz)

        else: st.error("--- فشل خط أنابيب البيانات في إحدى الخطوات ---")

    # --- مرحلة 5: عرض لوحة التحكم (عرض جميع الرسوم) ---
    def display_dashboard(self):
        """عرض جميع الرسوم البيانية المحسوبة."""
        st.title('📊 لوحة تحليل المخزون والمبيعات والتنبؤات')

        # استخدام .get مع قيمة افتراضية go.Figure() لعرض رسم فارغ إذا لم يتم إنشاؤه
        def get_fig(fig_key, title="N/A"):
            fig = self.visualizations.get(fig_key)
            if fig is None:
                 fig = go.Figure().update_layout(title=f"{title}: بيانات غير متوفرة").add_annotation(text="بيانات غير متوفرة", showarrow=False)
            return fig

        # --- تنظيم عرض الرسوم البيانية (كما في الكود القديم) ---
        st.header("تحليل عام للمخزون والكفاءة")
        col1, col2, col3 = st.columns(3)
        with col1: st.plotly_chart(get_fig('fig1', 'Vis 1'), use_container_width=True)
        with col2: st.plotly_chart(get_fig('fig2', 'Vis 2'), use_container_width=True)
        with col3: st.plotly_chart(get_fig('fig5', 'Vis 5'), use_container_width=True)

        st.divider()
        st.header("تحليل الإيرادات والمخزون")
        col4, col5 = st.columns(2)
        with col4: st.plotly_chart(get_fig('fig3', 'Vis 3'), use_container_width=True)
        with col5: st.plotly_chart(get_fig('fig4', 'Vis 4'), use_container_width=True)

        st.divider()
        st.header("تحليل باريتو والتسعير")
        col6, col7 = st.columns(2)
        with col6: st.plotly_chart(get_fig('fig6', 'Vis 6'), use_container_width=True)
        with col7: st.plotly_chart(get_fig('fig7', 'Vis 7'), use_container_width=True)

        st.divider()
        st.header("حالة المخزون (نقص وركود)")
        col8, col9 = st.columns(2)
        with col8: st.plotly_chart(get_fig('fig9', 'Vis 9'), use_container_width=True)
        with col9: st.plotly_chart(get_fig('fig10', 'Vis 10'), use_container_width=True)

        st.divider()
        st.header("المبيعات اليومية ومتوسط الفاتورة")
        col10, col11_placeholder = st.columns(2)
        with col10: st.plotly_chart(get_fig('fig11', 'Vis 11'), use_container_width=True) # fig11 الجديد
        with col11_placeholder: st.plotly_chart(get_fig('fig12', 'Vis 12'), use_container_width=True)

        st.divider()
        st.header("المنتجات الأقل والأكثر مبيعًا")
        col12, col13 = st.columns(2)
        with col12: st.plotly_chart(get_fig('fig13', 'Vis 13'), use_container_width=True)
        with col13: st.plotly_chart(get_fig('fig15', 'Vis 15'), use_container_width=True)

        st.divider()
        st.header("تكرار المبيعات ونسب الربح والإيراد")
        col14, col15 = st.columns(2)
        with col14:
             fig14_title = self.visualizations.get('fig14_title', "جدول تكرار المبيعات")
             st.subheader(fig14_title)
             fig14_obj = self.visualizations.get('fig14') # قد يكون None
             if fig14_obj: st.plotly_chart(fig14_obj, use_container_width=True)
             else: st.warning("لم يتم إنشاء جدول تكرار المبيعات (Vis 14).")
        with col15: st.plotly_chart(get_fig('fig16', 'Vis 16'), use_container_width=True)

        st.divider()
        st.header("المبالغ الآجلة للموردين")
        st.plotly_chart(get_fig('fig17', 'Vis 17'), use_container_width=True)

        # --- جدول المخزون الحالي ---
        st.divider()
        st.header("📦 المخزون الحالي")
        products_df_disp = self.processed_data.get('products', pd.DataFrame())
        product_flow_df_disp = self.analytics.get('product_flow', pd.DataFrame())

        if not products_df_disp.empty and 'id' in products_df_disp.columns and 'name' in products_df_disp.columns and 'quantity' in products_df_disp.columns:
            current_stock_display = products_df_disp[['id', 'name', 'quantity']].rename(columns={'quantity': 'current_stock', 'id': 'product_id'})
            if not product_flow_df_disp.empty and 'product_id' in product_flow_df_disp.columns and 'sales_quantity' in product_flow_df_disp.columns:
                 current_stock_display = current_stock_display.merge(product_flow_df_disp[['product_id', 'sales_quantity']], on='product_id', how='left').fillna({'sales_quantity': 0})

            display_cols_table = ['product_id', 'name', 'current_stock']
            if all(col in current_stock_display.columns for col in display_cols_table):
                try:
                    df_to_display = current_stock_display[display_cols_table].copy()
                    if 'product_id' in df_to_display.columns: df_to_display['product_id'] = df_to_display['product_id'].astype(str)
                    if 'name' in df_to_display.columns: df_to_display['name'] = df_to_display['name'].astype(str)
                    st.dataframe(
                        df_to_display.sort_values('current_stock', ascending=False).style.background_gradient(subset=['current_stock'], cmap='Greens'),
                        height=400, use_container_width=True,
                        column_config={"product_id": "ID", "name": "المنتج", "current_stock": st.column_config.NumberColumn("المخزون", format="%.1f")}
                    )
                except Exception as e_table_display:
                     st.error(f"خطأ عرض جدول المخزون: {e_table_display}")
            else: st.warning("أعمدة جدول المخزون غير متوفرة.")
        else: st.warning("لا توجد بيانات منتجات لعرض جدول المخزون.")


# --- التشغيل الرئيسي للداشبورد ---
if __name__ == "__main__":
    try:
        pipeline = DataPipeline()
        pipeline.run_pipeline()
        pipeline.display_dashboard()
    except FileNotFoundError as fnf_error:
        st.error(f"خطأ فادح: ملف أساسي مفقود: {fnf_error}")
        st.info("تأكد من مسارات ملفات Excel في بداية الكود.")
    except ImportError as imp_error:
        st.error(f"خطأ فادح: مكتبة أو ملف كود مفقود: {imp_error}")
        st.info("تأكد من تثبيت المكتبات ووجود ملفي feature_engineering.py و forecasting.py.")
    except Exception as e:
        st.error(f"حدث خطأ عام غير متوقع: {e}")
        st.error("تفاصيل الخطأ:")
        st.exception(e)

# --- End of Part 3 ---
