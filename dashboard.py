# dashboard.py - الكود الكامل مع مسار ثابت لملف CSV للتنبؤات (مُصحح)
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

# --- إعدادات مسارات البيانات الأصلية (Excel) ---
SALE_INVOICES_PATH = "data/sale_invoices.xlsx"
SALE_INVOICES_DETAILS_PATH = "data/sale_invoices_details.xlsx"
PRODUCTS_PATH = "data/products.xlsx"
INVOICE_DEFERRED_PATH = "data/invoice_deferreds.xlsx"

# --- !!! هام: تحديد مسار ملف CSV للتنبؤات هنا !!! ---
# *** غير هذا المسار ليشير إلى ملف CSV الخاص بك ***
FORECAST_CSV_PATH = "future_forecasts_standalone" # مثال: "C:/Users/YourUser/Documents/forecast.csv" أو "forecast_data.csv"
# ----------------------------------------------------

# --- تعريف أسماء الأعمدة المتوقعة في ملف CSV للتنبؤ ---
DATE_COLUMN_CSV = 'sale_date'
FORECAST_COLUMN_CSV = 'daily_sales_predicted'
LOWER_CI_COLUMN_CSV = 'lower_ci'
UPPER_CI_COLUMN_CSV = 'upper_ci'

# --- دالة مساعدة لتحليل التواريخ من Excel ---
def parse_excel_date(date_val):
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
            'daily_sales_actual': pd.DataFrame(), 'product_flow': pd.DataFrame(),
            'outstanding_amounts': pd.DataFrame(), 'pareto_data': pd.DataFrame(),
            'pie_data': {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}},
            'stagnant_products': pd.DataFrame(),
        }
        self.visualizations = {}
        self.forecast_data_from_csv = pd.DataFrame()
        self.original_data_paths = {
            'products': PRODUCTS_PATH, 'sale_invoices': SALE_INVOICES_PATH,
            'sale_invoices_details': SALE_INVOICES_DETAILS_PATH, 'invoice_deferred': INVOICE_DEFERRED_PATH
        }

    @st.cache_data(show_spinner="جاري تحميل البيانات الأصلية (Excel)...")
    def load_original_data(_self):
        raw_data_loaded = {}
        all_files_found = True
        for name, path in _self.original_data_paths.items():
             if not os.path.exists(path): st.error(f"لم يتم العثور على ملف '{name}': {path}"); all_files_found = False; raw_data_loaded[name] = pd.DataFrame()
             else:
                 try:
                     dtype_map = {}
                     if name == 'sale_invoices_details': dtype_map = {'quantity':'float32', 'totalPrice':'float32', 'buyPrice':'float32', 'price':'float32', 'discountPrice':'float32', 'invoice_id':'Int64', 'product_id':'Int64'}
                     elif name == 'sale_invoices': dtype_map = {'id':'Int64', 'totalPrice':'float32', 'paidAmount':'float32', 'remainingAmount':'float32', 'user_id':'Int64'}
                     elif name == 'products': dtype_map = {'id':'Int64', 'quantity':'float32', 'buyPrice':'float32', 'salePrice':'float32'}
                     raw_data_loaded[name] = pd.read_excel(path, dtype=dtype_map)
                 except Exception as e: st.error(f"خطأ تحميل {name}: {e}"); raw_data_loaded[name] = pd.DataFrame(); all_files_found = False
        required_for_analysis = ['sale_invoices', 'sale_invoices_details']
        if 'products' not in raw_data_loaded or raw_data_loaded['products'].empty: st.warning("تحذير: ملف المنتجات مفقود أو فارغ. بعض التحليلات قد لا تعمل.")
        missing_essentials = [name for name in required_for_analysis if name not in raw_data_loaded or raw_data_loaded[name].empty]
        if missing_essentials: st.error(f"الملفات الأساسية للفواتير مفقودة: {', '.join(missing_essentials)}. لا يمكن المتابعة."); st.stop()
        return raw_data_loaded

    @st.cache_data(show_spinner="جاري معالجة البيانات الأصلية وتحليلها...")
    def preprocess_and_analyze(_self, raw_data):
        processed_data = {}; analytics = {'daily_sales_actual': pd.DataFrame(columns=['date', 'actual']),'product_flow': pd.DataFrame(), 'outstanding_amounts': pd.DataFrame(),'pareto_data': pd.DataFrame(), 'stagnant_products': pd.DataFrame(),'pie_data': {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}}}
        si_raw = raw_data.get('sale_invoices', pd.DataFrame()); sid_raw = raw_data.get('sale_invoices_details', pd.DataFrame()); products_raw = raw_data.get('products', pd.DataFrame()); deferred_raw = raw_data.get('invoice_deferred', pd.DataFrame())
        if si_raw.empty or sid_raw.empty: st.error("بيانات الفواتير أو التفاصيل فارغة."); return processed_data, analytics
        si = si_raw.copy(); sid = sid_raw.copy(); products = products_raw.copy(); deferred = deferred_raw.copy()
        si_date_col = next((col for col in ['created_at', 'date', 'invoice_date', 'تاريخ_الإنشاء', 'إنشئ في'] if col in si.columns), None); si_id_col = next((col for col in ['id', 'invoice_id', 'رقم_الفاتورة'] if col in si.columns), None); si_total_col = next((col for col in ['totalPrice', 'total_price', 'الإجمالي'] if col in si.columns), 'totalPrice')
        sid_invoice_id_col = next((col for col in ['invoice_id', 'InvoiceId', 'رقم_الفاتورة'] if col in sid.columns), None); sid_amount_col = next((col for col in ['totalPrice', 'total_price', 'amount', 'الإجمالي'] if col in sid.columns), None); sid_quantity_col = next((col for col in ['quantity', 'qty', 'الكمية'] if col in sid.columns), None); sid_buy_price_col = next((col for col in ['buyPrice', 'cost_price', 'سعر_الشراء'] if col in sid.columns), None); sid_product_id_col = next((col for col in ['product_id', 'item_id', 'معرف_المنتج'] if col in sid.columns), None); sid_created_at_col = next((col for col in ['created_at', 'إنشئ_في', 'التاريخ'] if col in sid.columns), None)
        prod_id_col = next((col for col in ['id', 'product_id'] if col in products.columns), None); prod_name_col = next((col for col in ['name', 'product_name'] if col in products.columns), None); prod_quantity_col = next((col for col in ['quantity', 'stock', 'المخزون'] if col in products.columns), None); prod_buy_price_col = next((col for col in ['buyPrice', 'cost'] if col in products.columns), None); prod_sale_price_col = next((col for col in ['salePrice', 'price'] if col in products.columns), None)
        required_missing = []; check_cols = {'si_date_col': si_date_col, 'si_id_col': si_id_col, 'sid_invoice_id_col': sid_invoice_id_col, 'sid_amount_col': sid_amount_col, 'sid_quantity_col': sid_quantity_col, 'sid_product_id_col': sid_product_id_col}
        for name, col in check_cols.items():
            if not col: required_missing.append(f"عمود مفقود: {name.replace('_col','')} ({'si' if 'si_' in name else 'sid'})")
        if products_raw.empty: required_missing.append("ملف المنتجات فارغ")
        else:
            if not prod_id_col: required_missing.append("عمود ID (products)");
            if not prod_name_col: required_missing.append("عمود الاسم (products)")
        if any(c.startswith("عمود") for c in required_missing): st.error("الأعمدة الأساسية للفواتير مفقودة: " + ", ".join(required_missing)); return processed_data, analytics
        elif required_missing: st.warning("بعض الأعمدة/الملفات مفقودة: " + ", ".join(required_missing))
        try:
            si['parsed_date'] = si[si_date_col].apply(parse_excel_date); si.dropna(subset=['parsed_date'], inplace=True); si.rename(columns={si_id_col: 'invoice_pk'}, inplace=True); si['invoice_pk'] = pd.to_numeric(si['invoice_pk'], errors='coerce').astype('Int64'); si.dropna(subset=['invoice_pk'], inplace=True)
            si_total_col_original = si_total_col; si[si_total_col_original] = pd.to_numeric(si.get(si_total_col_original), errors='coerce').fillna(0).astype('float32') # Use .get() for safety
            si['created_at_date'] = pd.to_datetime(si['parsed_date']); si['created_at_per_day'] = si['created_at_date'].dt.date; si['year'] = si['created_at_date'].dt.year.astype('Int16'); si['month'] = si['created_at_date'].dt.month.astype('Int8'); si['month_name'] = si['created_at_date'].dt.strftime("%B"); si['year_month'] = si['created_at_date'].dt.strftime("%Y-%m")
            sid.rename(columns={sid_invoice_id_col: 'invoice_fk', sid_product_id_col: 'product_id'}, inplace=True); sid['invoice_fk'] = pd.to_numeric(sid['invoice_fk'], errors='coerce').astype('Int64'); sid['product_id'] = pd.to_numeric(sid['product_id'], errors='coerce').astype('Int64')
            sid_amount_col_original = sid_amount_col; sid[sid_amount_col_original] = pd.to_numeric(sid.get(sid_amount_col_original), errors='coerce').fillna(0).astype('float32')
            sid_quantity_col_original = sid_quantity_col; sid[sid_quantity_col_original] = pd.to_numeric(sid.get(sid_quantity_col_original), errors='coerce').fillna(0).astype('float32')
            sid_buy_price_col_original = sid_buy_price_col; sid[sid_buy_price_col_original if sid_buy_price_col_original in sid else 'buyPrice'] = pd.to_numeric(sid.get(sid_buy_price_col_original), errors='coerce').fillna(0).astype('float32'); sid_buy_price_col_original = sid_buy_price_col_original if sid_buy_price_col_original in sid else 'buyPrice'
            sid_created_at_col_original = sid_created_at_col
            if sid_created_at_col_original and sid_created_at_col_original in sid.columns: sid['created_at_dt'] = sid[sid_created_at_col_original].apply(parse_excel_date); sid['created_at_dt'] = pd.to_datetime(sid['created_at_dt'])
            else: sid = pd.merge(sid, si[['invoice_pk', 'created_at_date']], left_on='invoice_fk', right_on='invoice_pk', how='left'); sid.rename(columns={'created_at_date':'created_at_dt'}, inplace=True)
            sid['created_at_dt'] = pd.to_datetime(sid['created_at_dt'], errors='coerce'); sid.dropna(subset=['invoice_fk', 'product_id'], inplace=True); sid['netProfit'] = sid[sid_amount_col_original] - (sid[sid_buy_price_col_original] * sid[sid_quantity_col_original])
            if not products.empty:
                products.rename(columns={prod_id_col: 'product_pk', prod_name_col: 'product_name'}, inplace=True); products['product_pk'] = pd.to_numeric(products['product_pk'], errors='coerce').astype('Int64')
                prod_quantity_col_original = prod_quantity_col; products[prod_quantity_col_original if prod_quantity_col_original in products else 'quantity'] = pd.to_numeric(products.get(prod_quantity_col_original), errors='coerce').fillna(0).astype('float32')
                prod_buy_price_col_p = prod_buy_price_col; products[prod_buy_price_col_p if prod_buy_price_col_p in products else 'buyPrice'] = pd.to_numeric(products.get(prod_buy_price_col_p), errors='coerce').fillna(0).astype('float32')
                prod_sale_price_col_p = prod_sale_price_col; products[prod_sale_price_col_p if prod_sale_price_col_p in products else 'salePrice'] = pd.to_numeric(products.get(prod_sale_price_col_p), errors='coerce').fillna(0).astype('float32')
                products.dropna(subset=['product_pk', 'product_name'], inplace=True); products['name'] = products['product_name']; products['id'] = products['product_pk']
        except Exception as e_process: st.error(f"خطأ معالجة البيانات: {e_process}"); st.exception(e_process); processed_data={'sale_invoices': si,'sale_invoices_details': sid,'products': products,'invoice_deferred': deferred}; return processed_data, analytics
        try:
            merged_sales_fig11 = pd.merge(sid[['invoice_fk', sid_amount_col_original]], si[['invoice_pk', 'created_at_per_day']], left_on='invoice_fk', right_on='invoice_pk', how='left'); merged_sales_fig11.dropna(subset=['created_at_per_day'], inplace=True)
            if not merged_sales_fig11.empty: daily_sales_actual_df = merged_sales_fig11.groupby('created_at_per_day')[sid_amount_col_original].sum().reset_index(); daily_sales_actual_df.rename(columns={'created_at_per_day': 'date', sid_amount_col_original: 'actual'}, inplace=True); daily_sales_actual_df['date'] = pd.to_datetime(daily_sales_actual_df['date']).dt.date; analytics['daily_sales_actual'] = daily_sales_actual_df.sort_values('date')
            else: analytics['daily_sales_actual'] = pd.DataFrame(columns=['date', 'actual'])
            if not products.empty and not sid.empty:
                sales_qty_agg = sid.groupby('product_id')[sid_quantity_col_original].sum().reset_index(); sales_total_agg = sid.groupby('product_id')[sid_amount_col_original].sum().reset_index(); product_flow_df = pd.merge(sales_qty_agg.rename(columns={sid_quantity_col_original: 'sales_quantity'}), sales_total_agg.rename(columns={sid_amount_col_original: 'sales_amount'}), on='product_id', how='outer').fillna(0)
                product_flow_df = pd.merge(product_flow_df, products[['id', 'name', 'buyPrice', 'salePrice', 'quantity']].rename(columns={'quantity': 'current_stock'}), left_on='product_id', right_on='id', how='left').fillna({'name':'Unknown', 'buyPrice':0, 'salePrice':0, 'current_stock':0})
                product_flow_df['efficiency_ratio'] = product_flow_df['current_stock'] / product_flow_df['sales_quantity'].replace(0, np.nan); product_flow_df['efficiency'] = pd.cut(product_flow_df['efficiency_ratio'], bins=[0, 0.8, 1.2, float('inf')], labels=['Undersupplied', 'Balanced', 'Oversupplied'], right=False); product_flow_df['efficiency'] = product_flow_df['efficiency'].cat.add_categories('No Sales/Stock').fillna('No Sales/Stock')
                if 'created_at_dt' in sid.columns and not sid['created_at_dt'].isnull().all():
                     last_sale = sid.groupby('product_id')['created_at_dt'].max().reset_index(); product_flow_df = pd.merge(product_flow_df, last_sale, on='product_id', how='left'); product_flow_df.rename(columns={'created_at_dt': 'last_sale_date'}, inplace=True); product_flow_df['last_sale_date'] = pd.to_datetime(product_flow_df['last_sale_date'], errors='coerce'); today_date = datetime.now().date(); today_ts = pd.Timestamp(today_date); mask_valid_date = product_flow_df['last_sale_date'].notna(); product_flow_df.loc[mask_valid_date, 'days_since_last_sale'] = (today_ts - product_flow_df.loc[mask_valid_date, 'last_sale_date'].dt.normalize()).dt.days; product_flow_df['days_since_last_sale'] = pd.to_numeric(product_flow_df['days_since_last_sale'], errors='coerce').fillna(9999).astype(int)
                else: product_flow_df['last_sale_date'] = pd.NaT; product_flow_df['days_since_last_sale'] = 9999
                product_profit = sid.groupby('product_id')['netProfit'].sum().reset_index(); product_flow_df = pd.merge(product_flow_df, product_profit, on='product_id', how='left').fillna({'netProfit': 0}); product_flow_df['product_name'] = product_flow_df['name']; analytics['product_flow'] = product_flow_df
            else: analytics['product_flow'] = pd.DataFrame()
            if not analytics['product_flow'].empty:
                sorted_products = analytics['product_flow'].sort_values('sales_quantity', ascending=False); total_sales_quantity = sorted_products['sales_quantity'].sum()
                if total_sales_quantity > 0: sorted_products['cumulative_percentage'] = sorted_products['sales_quantity'].cumsum() / total_sales_quantity * 100
                else: sorted_products['cumulative_percentage'] = 0
                sorted_products['category'] = (sorted_products['cumulative_percentage'].fillna(0) // 10) * 10; analytics['pareto_data'] = sorted_products[sorted_products['cumulative_percentage'] <= 80].copy()
            else: analytics['pareto_data'] = pd.DataFrame()
            if not analytics['product_flow'].empty:
                product_flow_for_pie = analytics['product_flow']; top_revenue = product_flow_for_pie.sort_values('sales_amount', ascending=False).head(10); top_profit = product_flow_for_pie.sort_values('netProfit', ascending=False).head(10)
                total_revenue = product_flow_for_pie['sales_amount'].sum(); total_profit = product_flow_for_pie['netProfit'].sum(); top_revenue_pie = top_revenue[['name', 'sales_amount']].rename(columns={'sales_amount':'totalPrice'}); top_profit_pie = top_profit[['name', 'netProfit']]
                remaining_revenue_val = max(0, total_revenue - top_revenue_pie['totalPrice'].sum()); remaining_profit_val = max(0, total_profit - top_profit_pie['netProfit'].sum()); remaining_revenue = pd.DataFrame([{'name': 'Other Products', 'totalPrice': remaining_revenue_val}]); remaining_profit = pd.DataFrame([{'name': 'Other Products', 'netProfit': remaining_profit_val}])
                pie_revenue_df = pd.concat([top_revenue_pie, remaining_revenue], ignore_index=True) if not top_revenue_pie.empty else remaining_revenue; pie_profit_df = pd.concat([top_profit_pie, remaining_profit], ignore_index=True) if not top_profit_pie.empty else remaining_profit
                color_map = {row['name']: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, row in top_revenue.iterrows()}; color_map['Other Products'] = 'gray'; analytics['pie_data'] = {'revenue': pie_revenue_df[pie_revenue_df['totalPrice'] > 0].copy(), 'profit': pie_profit_df[pie_profit_df['netProfit'] > 0].copy(), 'color_mapping': color_map}
            else: analytics['pie_data'] = {'revenue': pd.DataFrame(), 'profit': pd.DataFrame(), 'color_mapping': {}}
            if not analytics['product_flow'].empty and 'days_since_last_sale' in analytics['product_flow'].columns:
                 days_col = pd.to_numeric(analytics['product_flow']['days_since_last_sale'], errors='coerce'); stagnant = analytics['product_flow'][(days_col >= 90) & (days_col != 9999)].copy()
                 if not stagnant.empty: stagnant['days_since_last_sale_num'] = pd.to_numeric(stagnant['days_since_last_sale'], errors='coerce'); stagnant.dropna(subset=['days_since_last_sale_num'], inplace=True); stagnant['days_category'] = pd.cut(stagnant['days_since_last_sale_num'], bins=[90, 270, 365, float('inf')], labels=['3-9 أشهر', '9-12 شهر', 'أكثر من سنة'], right=False); analytics['stagnant_products'] = stagnant.sort_values('days_since_last_sale_num', ascending=False).drop(columns=['days_since_last_sale_num'])
                 else: pass
            else: analytics['stagnant_products'] = pd.DataFrame()
            if not deferred.empty:
                def_type_col = 'invoice_type'; def_status_col = 'status'; def_amount_col = 'amount'; def_paid_col = 'paid_amount'; def_user_id_col = 'user_id'; required_deferred_cols = [def_type_col, def_status_col, def_amount_col, def_paid_col, def_user_id_col]
                if all(col in deferred.columns for col in required_deferred_cols):
                    deferred[def_amount_col] = pd.to_numeric(deferred[def_amount_col], errors='coerce').fillna(0); deferred[def_paid_col] = pd.to_numeric(deferred[def_paid_col], errors='coerce').fillna(0); buy_invoice_type_str = "Stocks\\Models\\BuyInvoice"; status_values = [0, 2]; filtered = deferred[(deferred[def_type_col] == buy_invoice_type_str) & (deferred[def_status_col].isin(status_values))].copy()
                    if not filtered.empty: filtered["outstanding_amount"] = filtered[def_amount_col] - filtered[def_paid_col]; outstanding_grouped = filtered.groupby(def_user_id_col, as_index=False)["outstanding_amount"].sum(); analytics['outstanding_amounts'] = outstanding_grouped[outstanding_grouped['outstanding_amount'] > 0].copy()
                    else: pass
                else: missing_cols = [col for col in required_deferred_cols if col not in deferred.columns]; st.warning(f"أعمدة مفقودة في بيانات الدفع الآجل: {', '.join(missing_cols)}"); analytics['outstanding_amounts'] = pd.DataFrame()
            else: analytics['outstanding_amounts'] = pd.DataFrame()
        except Exception as e_analyze: st.error(f"خطأ حساب التحليلات: {e_analyze}"); st.exception(e_analyze)
        processed_data['sale_invoices'] = si; processed_data['sale_invoices_details'] = sid; processed_data['products'] = products; processed_data['invoice_deferred'] = deferred
        return processed_data, analytics

    def load_forecast_csv(self, file_path):
        if not file_path: st.warning("لم يحدد مسار ملف CSV للتنبؤات (FORECAST_CSV_PATH)."); self.forecast_data_from_csv = pd.DataFrame(); return False
        if not os.path.exists(file_path): st.error(f"ملف CSV للتنبؤات غير موجود: {file_path}"); self.forecast_data_from_csv = pd.DataFrame(); return False
        try:
            df = pd.read_csv(file_path); st.success(f"تم تحميل التنبؤات CSV: {file_path}")
            required_csv_cols = [DATE_COLUMN_CSV, FORECAST_COLUMN_CSV]; missing_required_cols = [col for col in required_csv_cols if col not in df.columns]
            if missing_required_cols: st.error(f"ملف CSV '{file_path}' يفتقد للأعمدة: {', '.join(missing_required_cols)}"); return False
            try: df[DATE_COLUMN_CSV] = pd.to_datetime(df[DATE_COLUMN_CSV])
            except Exception as e_date: st.error(f"خطأ تحويل التاريخ '{DATE_COLUMN_CSV}' في CSV '{file_path}': {e_date}"); return False
            numeric_cols_csv = [FORECAST_COLUMN_CSV]; optional_csv_cols = [LOWER_CI_COLUMN_CSV, UPPER_CI_COLUMN_CSV]
            if LOWER_CI_COLUMN_CSV in df.columns: numeric_cols_csv.append(LOWER_CI_COLUMN_CSV)
            if UPPER_CI_COLUMN_CSV in df.columns: numeric_cols_csv.append(UPPER_CI_COLUMN_CSV)
            for col in numeric_cols_csv:
                if col in df.columns: original_type = df[col].dtype; df[col] = pd.to_numeric(df[col], errors='coerce');
                if df[col].isnull().any() and not pd.api.types.is_numeric_dtype(original_type): st.warning(f"تحذير (CSV): قيم غير رقمية في '{col}' بملف '{file_path}' -> NaN.")
            df = df.sort_values(by=DATE_COLUMN_CSV).reset_index(drop=True); self.forecast_data_from_csv = df; return True
        except Exception as e: st.error(f"خطأ قراءة/معالجة CSV '{file_path}': {e}"); self.forecast_data_from_csv = pd.DataFrame(); return False

    def generate_visualizations(self):
        self.visualizations = {}; product_flow = self.analytics.get('product_flow', pd.DataFrame()); sid_data = self.processed_data.get('sale_invoices_details', pd.DataFrame()); si_data = self.processed_data.get('sale_invoices', pd.DataFrame()); products_data = self.processed_data.get('products', pd.DataFrame())
        fig1_title="📊 مقارنة المخزون والمبيعات (Top 20)";fig2_title="📈 توزيع الكفاءة";fig3_title="💰 تحليل الإيرادات (مع COGS)";fig4_title="📦 المخزون الحالي مقابل المبيعات";fig5_title="⚖️ مقارنة الكفاءة";fig6_title="📊 تحليل باريتو (80/20)";fig7_title="📈 تحليل تسعير المنتجات";fig9_title="📦 منتجات تحتاج لإعادة تخزين (<10)";fig10_title='📅 المنتجات الراكدة (90+ يوم)';fig11_title="📈 المبيعات اليومية: الفعلية والمتوقعة";fig12_title="متوسط الفاتورة الشهرية";fig13_title="أقل 10 منتجات مبيعًا";fig14_title_display="📊 جدول تكرار كميات البيع";fig15_title="أكثر 10 منتجات مبيعًا";fig16_title="أعلى 10 منتجات ربحًا وإيرادًا";fig17_title="الموردين حسب المبلغ الآجل"
        if not product_flow.empty: top20_qty = product_flow.sort_values('sales_quantity', ascending=False).head(20); fig1 = go.Figure();
        if not top20_qty.empty: fig1.add_trace(go.Bar(x=top20_qty['name'], y=top20_qty['current_stock'], name='المخزون الحالي', marker_color='blue')); fig1.add_trace(go.Bar(x=top20_qty['name'], y=top20_qty['sales_quantity'], name='المبيعات', marker_color='orange')); fig1.update_layout(title=fig1_title, xaxis_title="المنتج", yaxis_title="الكمية", barmode="group")
        else: fig1.update_layout(title=fig1_title).add_annotation(text="لا توجد بيانات", showarrow=False); self.visualizations['fig1'] = fig1
        else: self.visualizations['fig1'] = go.Figure().update_layout(title=fig1_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        if not product_flow.empty and 'efficiency' in product_flow.columns: valid_efficiency_cats = ['Undersupplied', 'Balanced', 'Oversupplied']; efficiency_counts = product_flow[product_flow['efficiency'].isin(valid_efficiency_cats)]['efficiency'].value_counts();
        if not efficiency_counts.empty: fig2 = px.pie(values=efficiency_counts.values, names=efficiency_counts.index, title=fig2_title, color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']); self.visualizations['fig2'] = fig2
        else: self.visualizations['fig2'] = go.Figure().update_layout(title=fig2_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig2'] = go.Figure().update_layout(title=fig2_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        if not product_flow.empty and all(c in product_flow.columns for c in ['buyPrice', 'sales_quantity', 'sales_amount', 'name']): top20_amount = product_flow.sort_values('sales_amount', ascending=False).head(20).copy();
        if not top20_amount.empty: top20_amount['COGS'] = top20_amount['sales_quantity'] * top20_amount['buyPrice']; top20_amount['margin'] = top20_amount['sales_amount'] - top20_amount['COGS']; fig3 = px.bar(top20_amount, x="name", y=["COGS", "sales_amount", "margin"], barmode="group", title=fig3_title, labels={'value': 'المبلغ', 'variable': 'النوع', 'name':'المنتج'}, color_discrete_map={"COGS": "blue", "sales_amount": "orange", "margin": "green"}); self.visualizations['fig3'] = fig3
        else: self.visualizations['fig3'] = go.Figure().update_layout(title=fig3_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig3'] = go.Figure().update_layout(title=fig3_title).add_annotation(text="بيانات غير كافية", showarrow=False)
        if not product_flow.empty:
            if 'fig1' in self.visualizations and self.visualizations['fig1'].data and hasattr(self.visualizations['fig1'].data[0], 'x'): top20_names_fig1 = self.visualizations['fig1'].data[0].x; top20_qty_fig4 = product_flow[product_flow['name'].isin(top20_names_fig1)].nlargest(20, 'sales_quantity')
            else: top20_qty_fig4 = product_flow.nlargest(20, 'sales_quantity')
            if not top20_qty_fig4.empty: fig4 = go.Figure(); fig4.add_trace(go.Bar(x=top20_qty_fig4['name'], y=top20_qty_fig4['current_stock'], name="المخزون الحالي", marker_color='blue')); fig4.add_trace(go.Scatter(x=top20_qty_fig4['name'], y=top20_qty_fig4['sales_quantity'], mode='lines+markers', name="المبيعات", line=dict(color='orange'))); fig4.update_layout(title=fig4_title, xaxis_title="المنتج", yaxis_title="الكمية", barmode="overlay"); self.visualizations['fig4'] = fig4
            else: self.visualizations['fig4'] = go.Figure().update_layout(title=fig4_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig4'] = go.Figure().update_layout(title=fig4_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        if not product_flow.empty and 'efficiency' in product_flow.columns: valid_cats = ['Undersupplied', 'Balanced', 'Oversupplied']; eff_group = product_flow[product_flow['efficiency'].isin(valid_cats)].groupby('efficiency', observed=False)[['current_stock', 'sales_quantity']].sum().reset_index();
        if not eff_group.empty: fig5 = go.Figure(); fig5.add_trace(go.Bar(x=eff_group['efficiency'], y=eff_group['current_stock'], name='المخزون الحالي', marker_color='blue', opacity=0.6)); fig5.add_trace(go.Bar(x=eff_group['efficiency'], y=eff_group['sales_quantity'], name='المبيعات', marker_color='orange', opacity=0.6)); fig5.update_layout(barmode='group', title=fig5_title, xaxis_title="الفئة", yaxis_title="الكمية"); self.visualizations['fig5'] = fig5
        else: self.visualizations['fig5'] = go.Figure().update_layout(title=fig5_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig5'] = go.Figure().update_layout(title=fig5_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        pareto_data = self.analytics.get('pareto_data', pd.DataFrame()); fig6 = go.Figure();
        if not pareto_data.empty and all(c in pareto_data.columns for c in ['name', 'sales_quantity', 'cumulative_percentage', 'category']): fig6.add_trace(go.Bar(x=pareto_data['name'], y=pareto_data['sales_quantity'], name="المبيعات", marker_color='blue')); colors = px.colors.qualitative.Plotly; valid_categories = sorted([cat for cat in pareto_data['category'].unique() if pd.notna(cat)]);
        for i, category in enumerate(valid_categories): cat_data = pareto_data[pareto_data['category'] == category];
        if not cat_data.empty: fig6.add_trace(go.Scatter(x=cat_data['name'], y=cat_data['cumulative_percentage'], name=f"{int(category)}-{int(category + 10)}%", mode='lines+markers', line=dict(color=colors[i % len(colors)], dash='dash'), yaxis="y2")); fig6.update_layout(title=fig6_title, xaxis_title="المنتج", yaxis_title="المبيعات", yaxis2=dict(title="النسبة التراكمية (%)", overlaying="y", side="right", range=[0, 100], showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); self.visualizations['fig6'] = fig6
        else: self.visualizations['fig6'] = go.Figure().update_layout(title=fig6_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        if not product_flow.empty and all(c in product_flow.columns for c in ['salePrice', 'sales_quantity', 'name']): fig7_data = product_flow[(product_flow['salePrice'] > 0) & (product_flow['sales_quantity'] > 0)];
        if not fig7_data.empty: fig7 = px.scatter(fig7_data, x="salePrice", y="sales_quantity", title=fig7_title, labels={'salePrice': 'سعر البيع', 'sales_quantity': 'حجم المبيعات'}, trendline="lowess", hover_data=['name']); self.visualizations['fig7'] = fig7
        else: self.visualizations['fig7'] = go.Figure().update_layout(title=fig7_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig7'] = go.Figure().update_layout(title=fig7_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        if not product_flow.empty and all(c in product_flow.columns for c in ['current_stock', 'name']): restock = product_flow[product_flow['current_stock'] <= 10].sort_values('current_stock');
        if not restock.empty: fig9 = px.bar(restock, x="name", y="current_stock", title=fig9_title, labels={'name': 'المنتج', 'current_stock': 'المخزون الحالي'}); fig9.update_layout(xaxis={'categoryorder':'total ascending'}); self.visualizations['fig9'] = fig9
        else: self.visualizations['fig9'] = go.Figure().update_layout(title=fig9_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig9'] = go.Figure().update_layout(title=fig9_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        stagnant_products_orig = self.analytics.get('stagnant_products', pd.DataFrame());
        if not stagnant_products_orig.empty: stagnant_products_for_plot = stagnant_products_orig.copy(); hover_data_dict_str = {};
        if 'product_id' in stagnant_products_for_plot.columns: stagnant_products_for_plot['product_id_str'] = stagnant_products_for_plot['product_id'].astype(str).fillna('N/A'); hover_data_dict_str['ID'] = stagnant_products_for_plot['product_id_str'];
        if 'last_sale_date' in stagnant_products_for_plot.columns: stagnant_products_for_plot['last_sale_date'] = pd.to_datetime(stagnant_products_for_plot['last_sale_date'], errors='coerce'); stagnant_products_for_plot['last_sale_date_str'] = stagnant_products_for_plot['last_sale_date'].dt.strftime('%Y-%m-%d').fillna('N/A'); hover_data_dict_str['آخر بيع'] = stagnant_products_for_plot['last_sale_date_str'];
        if 'current_stock' in stagnant_products_for_plot.columns: stagnant_products_for_plot['current_stock_str'] = stagnant_products_for_plot['current_stock'].round(1).astype(str).fillna('N/A'); hover_data_dict_str['المخزون'] = stagnant_products_for_plot['current_stock_str']; required_cols_fig10 = ['name', 'days_since_last_sale', 'days_category'];
        if all(c in stagnant_products_for_plot.columns for c in required_cols_fig10):
            try: stagnant_products_for_plot['days_since_last_sale'] = pd.to_numeric(stagnant_products_for_plot['days_since_last_sale'], errors='coerce'); stagnant_products_for_plot.dropna(subset=['days_since_last_sale'], inplace=True);
            if not stagnant_products_for_plot.empty: fig10 = px.bar(stagnant_products_for_plot, x='name', y='days_since_last_sale', color='days_category', title=fig10_title, labels={'name': 'المنتج', 'days_since_last_sale': 'أيام منذ آخر بيع', 'days_category': 'فترة الركود'}, color_discrete_map={'3-9 أشهر': '#0000FF', '9-12 شهر': '#FF6347', 'أكثر من سنة': '#DC143C'}, hover_data=hover_data_dict_str if hover_data_dict_str else None); fig10.update_layout(xaxis_title="المنتج", yaxis_title="أيام منذ آخر بيع", xaxis={'categoryorder':'total descending'}, hovermode="x unified"); fig10.add_hline(y=180, line_dash="dot", annotation_text="حد 6 أشهر", annotation_position="top right", line_color="red"); self.visualizations['fig10'] = fig10
            else: self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="لا توجد بيانات", showarrow=False)
            except Exception as e_fig10: st.error(f"خطأ رسم 10: {e_fig10}"); self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="خطأ إنشاء", showarrow=False)
        else: self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="بيانات غير كافية", showarrow=False);
        else: self.visualizations['fig10'] = go.Figure().update_layout(title=fig10_title).add_annotation(text="لا توجد بيانات", showarrow=False)
        daily_sales_actual_df_fig11 = self.analytics.get('daily_sales_actual', pd.DataFrame()); future_forecast_df_fig11 = self.forecast_data_from_csv; fig11 = go.Figure(); last_actual_point_details_fig11 = None;
        if isinstance(daily_sales_actual_df_fig11, pd.DataFrame) and not daily_sales_actual_df_fig11.empty:
            try: df_actual_plot = daily_sales_actual_df_fig11.copy(); df_actual_plot['date'] = pd.to_datetime(df_actual_plot['date']); df_actual_plot = df_actual_plot.sort_values('date');
            if not df_actual_plot.empty: last_actual_row = df_actual_plot.iloc[-1]; last_actual_point_details_fig11 = {'date': last_actual_row['date'], 'actual': last_actual_row['actual']}; fig11.add_trace(go.Scatter(x=df_actual_plot['date'], y=df_actual_plot['actual'], mode='lines+markers', name='المبيعات الفعلية', marker=dict(color='rgba(0, 116, 217, 0.8)', size=5), line=dict(color='rgba(0, 116, 217, 0.8)', width=2)))
            except Exception as e_plot_actual_fig11: st.warning(f"خطأ رسم فعلية fig11: {e_plot_actual_fig11}"); last_actual_point_details_fig11 = None
        if isinstance(future_forecast_df_fig11, pd.DataFrame) and not future_forecast_df_fig11.empty: forecast_to_plot = future_forecast_df_fig11.copy();
        if DATE_COLUMN_CSV in forecast_to_plot.columns: forecast_to_plot[DATE_COLUMN_CSV] = pd.to_datetime(forecast_to_plot[DATE_COLUMN_CSV])
        else: st.warning(f"عمود '{DATE_COLUMN_CSV}' مفقود CSV."); forecast_to_plot = pd.DataFrame();
        if FORECAST_COLUMN_CSV in forecast_to_plot.columns and not forecast_to_plot.empty:
            try: forecast_to_plot = forecast_to_plot.sort_values(DATE_COLUMN_CSV); connected_forecast_df = forecast_to_plot;
            if last_actual_point_details_fig11 is not None: last_actual_dt = last_actual_point_details_fig11['date']; last_actual_val = last_actual_point_details_fig11['actual']; forecast_future_only = forecast_to_plot[forecast_to_plot[DATE_COLUMN_CSV] > last_actual_dt].copy(); connection_point_data = {DATE_COLUMN_CSV: last_actual_dt, FORECAST_COLUMN_CSV: last_actual_val, LOWER_CI_COLUMN_CSV: np.nan if LOWER_CI_COLUMN_CSV in forecast_future_only.columns else None, UPPER_CI_COLUMN_CSV: np.nan if UPPER_CI_COLUMN_CSV in forecast_future_only.columns else None}; connection_point_data = {k: v for k, v in connection_point_data.items() if k in forecast_future_only.columns or k == DATE_COLUMN_CSV}; connection_df = pd.DataFrame([connection_point_data]);
            if not forecast_future_only.empty: connected_forecast_df = pd.concat([connection_df, forecast_future_only], ignore_index=True)
            else: connected_forecast_df = pd.DataFrame();
            if not connected_forecast_df.empty: fig11.add_trace(go.Scatter(x=connected_forecast_df[DATE_COLUMN_CSV], y=connected_forecast_df[FORECAST_COLUMN_CSV], mode='lines', name='المبيعات المتوقعة', line=dict(color='rgba(255, 65, 54, 0.9)', dash='dash', width=2))); has_lower_ci_csv = LOWER_CI_COLUMN_CSV in connected_forecast_df.columns; has_upper_ci_csv = UPPER_CI_COLUMN_CSV in connected_forecast_df.columns;
            if has_lower_ci_csv and has_upper_ci_csv: ci_valid_csv = connected_forecast_df.iloc[1:].dropna(subset=[LOWER_CI_COLUMN_CSV, UPPER_CI_COLUMN_CSV]);
            if not ci_valid_csv.empty: fig11.add_trace(go.Scatter(x=ci_valid_csv[DATE_COLUMN_CSV].tolist() + ci_valid_csv[DATE_COLUMN_CSV].tolist()[::-1], y=ci_valid_csv[UPPER_CI_COLUMN_CSV].tolist() + ci_valid_csv[LOWER_CI_COLUMN_CSV].tolist()[::-1], fill='toself', fillcolor='rgba(255, 65, 54, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='فترة الثقة 95%', showlegend=True, hoverinfo='skip'));
            except Exception as e_plot_forecast_fig11: st.warning(f"خطأ رسم تنبؤ fig11: {e_plot_forecast_fig11}")
        elif not future_forecast_df_fig11.empty: st.warning(f"ملف CSV يفتقد لعمود '{FORECAST_COLUMN_CSV}'.")
        elif last_actual_point_details_fig11 is not None: st.info("لم يتم تحميل بيانات التنبؤ CSV.")
        fig11.update_layout(title=fig11_title, xaxis_title="التاريخ", yaxis_title="إجمالي المبيعات", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)); all_dates_fig11 = pd.Series(dtype='datetime64[ns]');
        if isinstance(daily_sales_actual_df_fig11, pd.DataFrame) and not daily_sales_actual_df_fig11.empty: all_dates_fig11 = pd.concat([all_dates_fig11, pd.to_datetime(daily_sales_actual_df_fig11['date'])]);
        if isinstance(future_forecast_df_fig11, pd.DataFrame) and not future_forecast_df_fig11.empty and DATE_COLUMN_CSV in future_forecast_df_fig11.columns: all_dates_fig11 = pd.concat([all_dates_fig11, pd.to_datetime(future_forecast_df_fig11[DATE_COLUMN_CSV])]);
        if not all_dates_fig11.empty:
            try: min_date_fig11 = all_dates_fig11.min(); max_date_fig11 = all_dates_fig11.max(); delta_days_fig11 = (max_date_fig11 - min_date_fig11).days if max_date_fig11 > min_date_fig11 else 1; date_margin_fig11 = timedelta(days=max(7, delta_days_fig11 * 0.05)); fig11.update_xaxes(range=[min_date_fig11 - date_margin_fig11, max_date_fig11 + date_margin_fig11])
            except Exception as e_xaxis_fig11: st.warning(f"خطأ تحديد نطاق X لـ fig11: {e_xaxis_fig11}"); self.visualizations['fig11'] = fig11
        si_total_col_fig12 = next((col for col in ['totalPrice', 'total_price', 'الإجمالي'] if col in si_data.columns), 'totalPrice');
        if not si_data.empty and all(c in si_data.columns for c in ['month_name', si_total_col_fig12, 'year', 'month']): monthly_avg = si_data.groupby(["year","month", "month_name"], as_index=False)[si_total_col_fig12].mean().sort_values(['year', 'month']);
        if not monthly_avg.empty: monthly_avg['display_month'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month_name']; fig12 = px.line(monthly_avg, x="display_month", y=si_total_col_fig12, title=fig12_title, labels={"display_month": "الشهر", si_total_col_fig12: "متوسط قيمة الفاتورة"}, markers=True); fig12.update_xaxes(type='category'); self.visualizations['fig12'] = fig12
        else: self.visualizations['fig12'] = go.Figure().update_layout(title=fig12_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig12'] = go.Figure().update_layout(title=fig12_title).add_annotation(text="أعمدة مفقودة", showarrow=False)
        if not product_flow.empty and all(c in product_flow.columns for c in ['sales_quantity', 'name', 'product_id']): sold_products = product_flow[product_flow['sales_quantity'] > 0].copy();
        if not sold_products.empty: bottom10 = sold_products.sort_values('sales_quantity').head(10); fig13 = px.bar(bottom10, x="sales_quantity", y="name", orientation='h', title=fig13_title, labels={"name": "المنتج", "sales_quantity": "الكمية المباعة"}, color="sales_quantity", color_continuous_scale=px.colors.sequential.Viridis, hover_data=["product_id"]); fig13.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title="الكمية المباعة", yaxis_title="المنتج"); self.visualizations['fig13'] = fig13
        else: self.visualizations['fig13'] = go.Figure().update_layout(title=fig13_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig13'] = go.Figure().update_layout(title=fig13_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        sid_qty_col_fig14 = next((col for col in ['quantity', 'qty', 'الكمية'] if col in sid_data.columns), 'quantity');
        # --- بداية الكود المصحح لـ fig14 ---
        self.visualizations['fig14'] = None # Initialize as None
        self.visualizations['fig14_title'] = fig14_title_display
        if not sid_data.empty and all(c in sid_data.columns for c in ['product_id', sid_qty_col_fig14]):
             sales_per_product = sid_data.groupby("product_id")[sid_qty_col_fig14].sum()
             sales_per_product = sales_per_product[pd.to_numeric(sales_per_product, errors='coerce').notnull() & (sales_per_product > 0)]
             if not sales_per_product.empty:
                 sales_freq = sales_per_product.value_counts().reset_index()
                 if len(sales_freq.columns) == 2:
                      sales_freq.columns = ["الكمية المباعة", "عدد المنتجات"]
                      fig14_data = sales_freq.sort_values(by="عدد المنتجات", ascending=False).head(15)
                      if not fig14_data.empty:
                           fig14_data["الكمية المباعة"] = fig14_data["الكمية المباعة"].round(1).astype(str)
                           fig14_data["عدد المنتجات"] = fig14_data["عدد المنتجات"].astype(str)
                           try:
                               fig14 = ff.create_table(fig14_data, index=False)
                               self.visualizations['fig14'] = fig14 # Assign only if creation succeeds
                           except Exception as e_ff_fig14: st.error(f"خطأ إنشاء جدول fig14: {e_ff_fig14}")
        # --- نهاية الكود المصحح لـ fig14 ---
        if not product_flow.empty and all(c in product_flow.columns for c in ['sales_quantity', 'name', 'product_id']): top10 = product_flow.sort_values('sales_quantity', ascending=False).head(10);
        if not top10.empty and top10['sales_quantity'].iloc[0] > 0: fig15 = px.bar(top10, x="sales_quantity", y="name", orientation='h', title=fig15_title, labels={"name": "المنتج", "sales_quantity": "الكمية المباعة"}, color="sales_quantity", color_continuous_scale=px.colors.sequential.Viridis, hover_data=["product_id"]); fig15.update_layout(yaxis={"categoryorder": "total ascending"}, height=600, xaxis_title="الكمية المباعة", yaxis_title="المنتج"); self.visualizations['fig15'] = fig15
        else: self.visualizations['fig15'] = go.Figure().update_layout(title=fig15_title).add_annotation(text="لا توجد بيانات", showarrow=False);
        else: self.visualizations['fig15'] = go.Figure().update_layout(title=fig15_title).add_annotation(text="بيانات مفقودة", showarrow=False)
        pie_data = self.analytics.get('pie_data');
        if pie_data and isinstance(pie_data, dict) and 'revenue' in pie_data and not pie_data['revenue'].empty and 'profit' in pie_data and not pie_data['profit'].empty and 'color_mapping' in pie_data and isinstance(pie_data['color_mapping'], dict): fig16 = make_subplots(rows=1, cols=2, subplot_titles=["الأعلى إيرادًا", "الأعلى ربحًا"], specs=[[{"type": "domain"}, {"type": "domain"}]]); revenue_to_plot = pie_data['revenue']; profit_to_plot = pie_data['profit'];
        if not revenue_to_plot.empty: revenue_colors = [pie_data['color_mapping'].get(n, "gray") for n in revenue_to_plot["name"]]; fig16.add_trace(go.Pie(labels=revenue_to_plot["name"], values=revenue_to_plot["totalPrice"], name="إيراد", marker=dict(colors=revenue_colors)), row=1, col=1)
        else: fig16.add_annotation(text="لا يوجد إيراد", xref="paper", yref="paper", x=0.18, y=0.5, showarrow=False);
        if not profit_to_plot.empty: profit_colors = [pie_data['color_mapping'].get(n, "gray") for n in profit_to_plot["name"]]; fig16.add_trace(go.Pie(labels=profit_to_plot["name"], values=profit_to_plot["netProfit"], name="ربح", marker=dict(colors=profit_colors)), row=1, col=2)
        else: fig16.add_annotation(text="لا يوجد ربح", xref="paper", yref="paper", x=0.82, y=0.5, showarrow=False); fig16.update_traces(hoverinfo='label+percent+value', textinfo='percent', textfont_size=11, insidetextorientation='radial'); fig16.update_layout(height=600, showlegend=False, title_text=fig16_title); self.visualizations['fig16'] = fig16
        else: self.visualizations['fig16'] = go.Figure().update_layout(title=fig16_title).add_annotation(text="بيانات غير كافية", showarrow=False)
        outstanding = self.analytics.get('outstanding_amounts', pd.DataFrame());
        if not outstanding.empty and all(c in outstanding.columns for c in ['user_id', 'outstanding_amount']): total_outstanding = outstanding["outstanding_amount"].sum(); outstanding_sorted = outstanding.sort_values("outstanding_amount", ascending=False); outstanding_sorted['user_id'] = outstanding_sorted['user_id'].astype(str); fig17 = px.bar(outstanding_sorted, x="user_id", y="outstanding_amount", color="outstanding_amount", title=fig17_title, labels={"user_id": "المورد", "outstanding_amount": "المبلغ الآجل"}, color_continuous_scale="Inferno"); fig17.update_coloraxes(colorbar_title="المبلغ الآجل"); fig17.update_layout(xaxis={'categoryorder': 'total descending', 'type': 'category'}); fig17.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text=f"مجموع الآجل: {round(total_outstanding,2)}", showarrow=False, font=dict(size=18, color="black"), bgcolor="rgba(211,211,211,0.7)", bordercolor="black", borderwidth=1, borderpad=4, align="right"); self.visualizations['fig17'] = fig17
        else: self.visualizations['fig17'] = go.Figure().update_layout(title=fig17_title).add_annotation(text="لا توجد بيانات", showarrow=False)

    def run_pipeline(self, forecast_csv_path):
        pipeline_success = True; excel_data_loaded = False; csv_data_loaded = False
        try: self.raw_data = self.load_original_data();
        if self.raw_data is None or not isinstance(self.raw_data, dict) or 'sale_invoices' not in self.raw_data or self.raw_data['sale_invoices'].empty or 'sale_invoices_details' not in self.raw_data or self.raw_data['sale_invoices_details'].empty: st.error("فشل تحميل Excel."); pipeline_success = False
        else: excel_data_loaded = True
        except Exception as e_load: st.error(f"فشل تحميل Excel: {e_load}"); st.exception(e_load); pipeline_success = False
        if pipeline_success:
             try: csv_data_loaded = self.load_forecast_csv(forecast_csv_path)
             except Exception as e_csv_load: st.error(f"خطأ تحميل CSV: {e_csv_load}"); st.exception(e_csv_load); self.forecast_data_from_csv = pd.DataFrame()
        if pipeline_success and excel_data_loaded:
             try: with st.spinner("جاري تحليل Excel..."): processed_data_result, analytics_result = self.preprocess_and_analyze(self.raw_data); self.processed_data = processed_data_result; self.analytics = analytics_result;
             is_analytics_valid = isinstance(self.analytics, dict) and self.analytics.get('daily_sales_actual') is not None and not self.analytics['daily_sales_actual'].empty;
             if not is_analytics_valid: st.warning("لم يتم حساب المبيعات الفعلية.")
             except Exception as e_analyze_pipe: st.error(f"فشل تحليل Excel: {e_analyze_pipe}"); st.exception(e_analyze_pipe); pipeline_success = False
        elif not excel_data_loaded: st.error("لا يمكن التحليل (فشل تحميل Excel)."); pipeline_success = False
        if pipeline_success:
             try:
                 if isinstance(self.analytics, dict) and self.analytics: with st.spinner("جاري إنشاء الرسوم..."): self.generate_visualizations();
                 if not self.visualizations: st.warning("لم يتم إنشاء رسوم.")
                 else: st.warning("لا يمكن إنشاء الرسوم (لا توجد بيانات تحليل).")
             except Exception as e_viz_pipe: st.error(f"فشل إنشاء الرسوم: {e_viz_pipe}"); st.exception(e_viz_pipe)
        else: st.error("--- فشل خط الأنابيب ---")

    def display_dashboard(self):
        st.title('📊 لوحة تحليل المخزون والمبيعات والتنبؤات')
        def get_fig(fig_key, title="N/A"): fig = self.visualizations.get(fig_key);
        if fig is None: fig = go.Figure().update_layout(title=f"{title}: بيانات غير متوفرة", xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{'text': "بيانات غير متوفرة", 'xref': "paper", 'yref': "paper",'showarrow': False, 'font': {'size': 16}}]); return fig
        st.header("تحليل عام للمخزون والكفاءة"); col1, col2, col3 = st.columns(3);
        with col1: st.plotly_chart(get_fig('fig1', 'مقارنة المخزون والمبيعات'), use_container_width=True)
        with col2: st.plotly_chart(get_fig('fig2', 'توزيع الكفاءة'), use_container_width=True)
        with col3: st.plotly_chart(get_fig('fig5', 'مقارنة الكفاءة'), use_container_width=True)
        st.divider(); st.header("تحليل الإيرادات والمخزون"); col4, col5 = st.columns(2);
        with col4: st.plotly_chart(get_fig('fig3', 'تحليل الإيرادات'), use_container_width=True)
        with col5: st.plotly_chart(get_fig('fig4', 'المخزون مقابل المبيعات'), use_container_width=True)
        st.divider(); st.header("تحليل باريتو والتسعير"); col6, col7 = st.columns(2);
        with col6: st.plotly_chart(get_fig('fig6', 'تحليل باريتو'), use_container_width=True)
        with col7: st.plotly_chart(get_fig('fig7', 'تحليل التسعير'), use_container_width=True)
        st.divider(); st.header("حالة المخزون (نقص وركود)"); col8, col9 = st.columns(2);
        with col8: st.plotly_chart(get_fig('fig9', 'منتجات تحتاج إعادة تخزين'), use_container_width=True)
        with col9: st.plotly_chart(get_fig('fig10', 'المنتجات الراكدة'), use_container_width=True)
        st.divider(); st.header("المبيعات اليومية ومتوسط الفاتورة"); col10, col11_placeholder = st.columns(2);
        with col10: st.plotly_chart(get_fig('fig11', 'المبيعات الفعلية والمتوقعة'), use_container_width=True)
        with col11_placeholder: st.plotly_chart(get_fig('fig12', 'متوسط الفاتورة الشهرية'), use_container_width=True)
        st.divider(); st.header("المنتجات الأقل والأكثر مبيعًا"); col12, col13 = st.columns(2);
        with col12: st.plotly_chart(get_fig('fig13', 'أقل المنتجات مبيعًا'), use_container_width=True)
        with col13: st.plotly_chart(get_fig('fig15', 'أكثر المنتجات مبيعًا'), use_container_width=True)
        st.divider(); st.header("تكرار المبيعات ونسب الربح والإيراد"); col14, col15 = st.columns(2);
        with col14: fig14_title = self.visualizations.get('fig14_title', "جدول تكرار كميات البيع"); st.subheader(fig14_title); fig14_obj = self.visualizations.get('fig14');
        if fig14_obj: st.plotly_chart(fig14_obj, use_container_width=True)
        else: st.warning("لم يتم إنشاء جدول تكرار المبيعات (Vis 14).")
        with col15: st.plotly_chart(get_fig('fig16', 'الأعلى ربحًا وإيرادًا'), use_container_width=True)
        st.divider(); st.header("المبالغ الآجلة للموردين"); st.plotly_chart(get_fig('fig17', 'المبالغ الآجلة للموردين'), use_container_width=True)
        st.divider(); st.header("📦 المخزون الحالي"); products_df_disp = self.processed_data.get('products', pd.DataFrame()); product_flow_df_disp = self.analytics.get('product_flow', pd.DataFrame());
        if not products_df_disp.empty and all(c in products_df_disp.columns for c in ['id', 'name', 'quantity']): current_stock_display = products_df_disp[['id', 'name', 'quantity']].rename(columns={'quantity': 'current_stock', 'id': 'product_id'});
        if not product_flow_df_disp.empty and all(c in product_flow_df_disp.columns for c in ['product_id', 'sales_quantity']): current_stock_display = current_stock_display.merge(product_flow_df_disp[['product_id', 'sales_quantity']], on='product_id', how='left').fillna({'sales_quantity': 0}); display_cols_table = ['product_id', 'name', 'current_stock'];
        if 'sales_quantity' in current_stock_display.columns: display_cols_table.append('sales_quantity');
        if all(col in current_stock_display.columns for col in display_cols_table):
            try: df_to_display = current_stock_display[display_cols_table].copy();
            if 'product_id' in df_to_display.columns: df_to_display['product_id'] = df_to_display['product_id'].astype(str);
            if 'name' in df_to_display.columns: df_to_display['name'] = df_to_display['name'].astype(str); column_config_dict = {"product_id": "ID", "name": "المنتج", "current_stock": st.column_config.NumberColumn("المخزون الحالي", format="%.1f")};
            if 'sales_quantity' in display_cols_table: column_config_dict["sales_quantity"] = st.column_config.NumberColumn("إجمالي المبيعات", format="%.1f"); st.dataframe(df_to_display.sort_values('current_stock', ascending=False).style.background_gradient(subset=['current_stock'], cmap='Greens'), height=400, use_container_width=True, column_config=column_config_dict)
            except Exception as e_table_display: st.error(f"خطأ عرض جدول المخزون: {e_table_display}")
        else: st.warning("أعمدة جدول المخزون غير متوفرة.")
        else: st.warning("لا توجد بيانات منتجات لعرض جدول المخزون.")

if __name__ == "__main__":
    st.sidebar.subheader("ملفات البيانات المستخدمة:")
    st.sidebar.caption(f"الفواتير Excel: `{SALE_INVOICES_PATH}`")
    st.sidebar.caption(f"التفاصيل Excel: `{SALE_INVOICES_DETAILS_PATH}`")
    st.sidebar.caption(f"المنتجات Excel: `{PRODUCTS_PATH}`")
    st.sidebar.caption(f"الآجل Excel: `{INVOICE_DEFERRED_PATH}`")
    st.sidebar.caption(f"التنبؤات CSV: `{FORECAST_CSV_PATH}`")
    st.sidebar.markdown("_تأكد من وجود الملفات وتعديل `FORECAST_CSV_PATH` في الكود._")
    pipeline = DataPipeline()
    try: pipeline.run_pipeline(forecast_csv_path=FORECAST_CSV_PATH); pipeline.display_dashboard()
    except FileNotFoundError as fnf_error: st.error(f"خطأ فادح: ملف Excel مفقود: {fnf_error}"); st.info("تحقق من مسارات Excel.")
    except Exception as e: st.error(f"خطأ عام غير متوقع: {e}"); st.error("تفاصيل الخطأ:"); st.exception(e)
