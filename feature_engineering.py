# feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import logging
import warnings
import os
# إعداد Logger الأساسي
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# تجاهل تحذير infer_datetime_format (اختياري)
warnings.filterwarnings("ignore", message="The argument 'infer_datetime_format' is deprecated")

# --- دالة parse_excel_date (مضمنة هنا لتجنب الاعتماد على dashboard.py) ---
# (مأخوذة من الكود الأصلي للداشبورد)
def _parse_excel_date_local(date_val):
    """يحاول تحليل التاريخ من الأرقام أو النصوص (نسخة محلية)."""
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
            # محاولة تنسيقات شائعة أولاً
            common_formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
            for fmt in common_formats:
                try:
                    return pd.to_datetime(date_val, format=fmt)
                except (ValueError, TypeError):
                    continue
            # إذا فشلت التنسيقات الشائعة، جرب التحليل العام
            return pd.to_datetime(date_val, errors='coerce')
        else:
            return pd.to_datetime(date_val, errors='coerce')
    except Exception:
        return pd.NaT

class _SalesFeatureEngineerInternal:
    def __init__(self, data_paths):
        self.logger = logging.getLogger(__name__)
        self.data = self._load_data(data_paths)
        self.features = None
        self.saudi_holidays = self._get_saudi_holidays()

    def _get_saudi_holidays(self):
        # نطاق ديناميكي للسنوات
        current_year = datetime.now().year
        years = range(max(2020, current_year - 3), current_year + 2) # 3 سنوات سابقة والسنة القادمة
        try:
            return holidays.SaudiArabia(years=years)
        except Exception as e:
            self.logger.warning(f"لم يتم تحميل الإجازات السعودية للسنوات {min(years)}-{max(years)}: {e}. سيتم استخدام قائمة فارغة.")
            return holidays.HolidayBase()

    def _load_data(self, paths):
        data_types = {
            # استخدام أنواع قابلة للقيم المفقودة Nullable
            'id': 'Int64', 'invoice_id': 'Int64', 'product_id': 'Int64', 'user_id': 'Int64',
            'quantity': 'float32', 'totalPrice': 'float32', 'price': 'float32',
            'discountPrice': 'float32', 'buyPrice': 'float32', 'paidAmount': 'float32',
            'remainingAmount': 'float32', 'amount': 'float32'
        }
        data = {}
        required_files = ['sale_invoices', 'sale_invoices_details']
        all_required_found = True

        for table_name, path in paths.items():
            try:
                if not os.path.exists(path):
                     self.logger.error(f"ملف '{table_name}' غير موجود في المسار: {path}")
                     if table_name in required_files: all_required_found = False
                     data[table_name] = pd.DataFrame()
                     continue

                # محاولة قراءة الأعمدة أولاً
                try:
                    df_cols = pd.read_excel(path, nrows=0).columns
                except Exception as e_read_cols:
                    self.logger.warning(f"لم نتمكن من قراءة أعمدة {table_name} مسبقًا: {e_read_cols}")
                    df_cols = []

                applicable_dtypes = {col: dtype for col, dtype in data_types.items() if col in df_cols}
                data[table_name] = pd.read_excel(path, dtype=applicable_dtypes)
                self.logger.info(f"تم تحميل {table_name} بنجاح ({len(data[table_name])} صف). الأعمدة: {data[table_name].columns.tolist()}")

            except Exception as e:
                self.logger.error(f"تعذر تحميل {table_name} من {path}: {str(e)}")
                if table_name in required_files: all_required_found = False
                data[table_name] = pd.DataFrame()

        if not all_required_found:
             raise FileNotFoundError("ملف الفواتير أو تفاصيلها الأساسية مفقود. لا يمكن إنشاء الميزات.")

        return data

    def _preprocess_dates(self):
        table = 'sale_invoices'
        if table not in self.data or self.data[table].empty:
            self.logger.warning(f"جدول '{table}' غير موجود أو فارغ، لا يمكن معالجة التواريخ.")
            # لا نرفع خطأ هنا، قد يتم التعامل معه لاحقًا
            return

        # البحث عن عمود التاريخ بمرونة
        date_col = next((col for col in ['created_at', 'إنشئ في', 'date', 'invoice_date', 'timestamp'] if col in self.data[table].columns), None)

        if not date_col:
            self.logger.error(f"لا يوجد عمود تاريخ معروف (مثل created_at, date) في جدول '{table}'.")
            raise ValueError(f"عمود التاريخ مفقود في '{table}'.") # فشل حاسم

        self.logger.info(f"معالجة عمود التاريخ '{date_col}' في جدول '{table}'. العدد الأولي للصفوف: {len(self.data[table])}")
        original_type = self.data[table][date_col].dtype
        self.logger.info(f"النوع الأصلي لعمود التاريخ '{date_col}': {original_type}")

        # استخدام الدالة المساعدة المحلية للتحليل
        self.data[table]['processed_date_dt'] = self.data[table][date_col].apply(_parse_excel_date_local)

        total_rows_before_drop = len(self.data[table])
        null_dates_count = self.data[table]['processed_date_dt'].isnull().sum()
        self.logger.info(f"بعد التحليل الأولي: إجمالي الصفوف = {total_rows_before_drop}, عدد التواريخ غير الصالحة (NaT) = {null_dates_count}")

        if null_dates_count == total_rows_before_drop:
             self.logger.error(f"فشل تحليل جميع التواريخ في العمود '{date_col}'. تحقق من تنسيقات البيانات.")
             raise ValueError(f"لم يتم العثور على تواريخ صالحة في '{table}'.")

        if null_dates_count > 0:
             # يمكنك طباعة عينة من القيم التي فشل تحليلها
             # invalid_rows_sample = self.data[table][self.data[table]['processed_date_dt'].isnull()][[date_col]].head()
             # self.logger.warning(f"عينة من القيم التي فشل تحليلها كتواريخ:\n{invalid_rows_sample}")
             self.data[table] = self.data[table].dropna(subset=['processed_date_dt'])
             self.logger.warning(f"تم حذف {null_dates_count} صفوف بسبب تواريخ غير صالحة (NaT). الصفوف المتبقية: {len(self.data[table])}")

        if self.data[table].empty:
             self.logger.error(f"لا توجد صفوف متبقية في '{table}' بعد معالجة التواريخ وحذف القيم غير الصالحة.")
             raise ValueError(f"لم يتم العثور على تواريخ صالحة في '{table}'.")

        # تحويل إلى تاريخ فقط
        self.data[table]['processed_date'] = self.data[table]['processed_date_dt'].dt.date
        unique_dates_after_processing = self.data[table]['processed_date'].nunique()
        self.logger.info(f"اكتملت معالجة التواريخ لـ '{table}'. عدد التواريخ الفريدة المتبقية: {unique_dates_after_processing}")
        # حذف العمود المؤقت
        self.data[table] = self.data[table].drop(columns=['processed_date_dt'])


    def _create_daily_sales(self):
        try:
            details_table = 'sale_invoices_details'
            invoices_table = 'sale_invoices'

            if details_table not in self.data or self.data[details_table].empty:
                raise ValueError(f"بيانات '{details_table}' غير متوفرة أو فارغة.")
            if invoices_table not in self.data or self.data[invoices_table].empty:
                # قد يكون هذا طبيعياً إذا تمت تصفية كل التواريخ في الخطوة السابقة
                 if 'processed_date' not in self.data.get(invoices_table, pd.DataFrame()).columns:
                      raise ValueError(f"جدول '{invoices_table}' فارغ أو يفتقد لعمود التاريخ المعالج.")
                 else: # الجدول موجود وبه عمود التاريخ لكنه فارغ
                      self.logger.warning(f"جدول الفواتير '{invoices_table}' أصبح فارغًا بعد معالجة التواريخ.")
                      # إرجاع DataFrame فارغ بالبنية الصحيحة
                      return pd.DataFrame(columns=['sale_date', 'daily_sales', 'transaction_count', 'total_items'])


            if 'processed_date' not in self.data[invoices_table].columns:
                 raise ValueError(f"عمود 'processed_date' مفقود في '{invoices_table}' قبل الدمج.")

            amount_col = next((col for col in ['totalPrice', 'total_price', 'amount'] if col in self.data[details_table].columns), None)
            if not amount_col: raise KeyError("لا يوجد عمود لإجمالي المبلغ في تفاصيل الفواتير.")
            self.logger.info(f"استخدام عمود '{amount_col}' لحساب المبيعات.")

            details_invoice_id_col = next((col for col in ['invoice_id', 'invoice_fk'] if col in self.data[details_table].columns), 'invoice_id')
            invoices_id_col = next((col for col in ['id', 'invoice_pk'] if col in self.data[invoices_table].columns), 'id')
            self.logger.info(f"ربط '{details_invoice_id_col}' من التفاصيل مع '{invoices_id_col}' من الفواتير.")

            qty_col = next((col for col in ['quantity', 'qty'] if col in self.data[details_table].columns), None)
            if not qty_col:
                self.logger.warning("عمود الكمية غير موجود، سيتم استخدام قيمة افتراضية 1 لحساب العناصر.")
                # إضافة عمود مؤقت إذا لم يكن موجودًا
                if 'quantity_default' not in self.data[details_table].columns:
                     self.data[details_table]['quantity_default'] = 1
                qty_col = 'quantity_default'
            else:
                 # التأكد من أن عمود الكمية رقمي
                 self.data[details_table][qty_col] = pd.to_numeric(self.data[details_table][qty_col], errors='coerce').fillna(0)


            # تحضير DataFrames للدمج
            df_details = self.data[details_table][[details_invoice_id_col, amount_col, qty_col]].copy()
            df_invoices = self.data[invoices_table][[invoices_id_col, 'processed_date']].copy()

            # التأكد من أنواع أعمدة الربط قبل الدمج
            df_details[details_invoice_id_col] = pd.to_numeric(df_details[details_invoice_id_col], errors='coerce').astype('Int64')
            df_invoices[invoices_id_col] = pd.to_numeric(df_invoices[invoices_id_col], errors='coerce').astype('Int64')

            rows_details_before_drop = len(df_details)
            rows_invoices_before_drop = len(df_invoices)
            # حذف القيم المفقودة في أعمدة الربط أو التاريخ الأساسي
            df_details.dropna(subset=[details_invoice_id_col], inplace=True)
            df_invoices.dropna(subset=[invoices_id_col, 'processed_date'], inplace=True)
            self.logger.info(f"قبل الدمج: صفوف التفاصيل = {len(df_details)} (من {rows_details_before_drop}), صفوف الفواتير = {len(df_invoices)} (من {rows_invoices_before_drop})")
            self.logger.info(f"قبل الدمج: الفواتير الفريدة (تفاصيل) = {df_details[details_invoice_id_col].nunique()}, (فواتير) = {df_invoices[invoices_id_col].nunique()}")
            self.logger.info(f"قبل الدمج: الأيام الفريدة (فواتير) = {df_invoices['processed_date'].nunique()}")


            # دمج البيانات (استخدام left join للحفاظ على كل تفاصيل الفواتير التي لها معرف صالح)
            merged = pd.merge(
                df_details,
                df_invoices,
                left_on=details_invoice_id_col,
                right_on=invoices_id_col,
                how='left' # Keep all details, match invoices
            )
            self.logger.info(f"بعد الدمج (how='left'): عدد الصفوف = {len(merged)}")

            # التحقق من الصفوف التي لم تجد فاتورة مطابقة أو تاريخ صالح
            null_dates_after_merge = merged['processed_date'].isnull().sum()
            if null_dates_after_merge > 0:
                 self.logger.warning(f"تم العثور على {null_dates_after_merge} تفاصيل فاتورة لم تجد فاتورة مطابقة أو تاريخ صالح. سيتم حذفها.")
                 merged.dropna(subset=['processed_date'], inplace=True)
                 self.logger.info(f"بعد حذف الصفوف غير المرتبطة بتاريخ: عدد الصفوف = {len(merged)}")

            if merged.empty:
                 self.logger.error("لا توجد بيانات صالحة بعد دمج الفواتير والتفاصيل وحذف غير المرتبط.")
                 # إرجاع DataFrame فارغ بالبنية الصحيحة
                 return pd.DataFrame(columns=['sale_date', 'daily_sales', 'transaction_count', 'total_items'])


            unique_days_before_grouping = merged['processed_date'].nunique()
            self.logger.info(f"عدد الأيام الفريدة *قبل* التجميع = {unique_days_before_grouping}")
            if unique_days_before_grouping < 3:
                 self.logger.warning(f"عدد الأيام الفريدة قبل التجميع هو {unique_days_before_grouping} وهو قليل جدًا!")

            # تجميع البيانات اليومية
            merged[amount_col] = pd.to_numeric(merged[amount_col], errors='coerce').fillna(0)
            # تأكد من أن عمود الكمية رقمي هنا أيضًا (احتياطي)
            merged[qty_col] = pd.to_numeric(merged[qty_col], errors='coerce').fillna(0)

            daily_sales = merged.groupby('processed_date').agg(
                daily_sales=(amount_col, 'sum'),
                transaction_count=(details_invoice_id_col, 'nunique'), # عدد الفواتير الفريدة في اليوم
                total_items=(qty_col, 'sum') # مجموع الكميات في اليوم
            ).reset_index()

            daily_sales = daily_sales.rename(columns={'processed_date': 'sale_date'})
            # التأكد من أن sale_date هو datetime object قبل إرجاعه
            daily_sales['sale_date'] = pd.to_datetime(daily_sales['sale_date'])

            num_days_final = len(daily_sales)
            self.logger.info(f"*** تم حساب المبيعات اليومية لـ {num_days_final} يوم. ***")
            if num_days_final == 0:
                 self.logger.error("فشل تجميع المبيعات اليومية، النتيجة فارغة.")
                 # هذا لا يجب أن يحدث إذا كان merged غير فارغ، لكنه تحقق إضافي
            elif num_days_final < 3:
                 self.logger.warning("عدد الأيام الناتجة بعد التجميع قليل جدًا!")

            return daily_sales.sort_values('sale_date')

        except (ValueError, KeyError, FileNotFoundError) as e:
            self.logger.error(f"خطأ مُعالج في إنشاء المبيعات اليومية: {str(e)}")
            raise # إعادة رفع الخطأ ليتم التقاطه في generate_features_df
        except Exception as e:
            self.logger.error(f"خطأ غير متوقع في إنشاء المبيعات اليومية: {str(e)}", exc_info=True)
            raise


    def _add_time_features(self, df):
        if df.empty or 'sale_date' not in df.columns:
            # self.logger.debug("DataFrame فارغ أو يفتقد 'sale_date' في _add_time_features.")
            return df

        # استخدام نسخة لتجنب SettingWithCopyWarning إذا تم تمرير شريحة
        df_out = df.copy()
        # التأكد من أن عمود التاريخ هو datetime
        df_out['date_temp'] = pd.to_datetime(df_out['sale_date'])

        try:
            df_out['day_of_week'] = df_out['date_temp'].dt.dayofweek.astype('int8')
            df_out['is_weekend'] = df_out['day_of_week'].isin([4, 5]).astype('int8')
            df_out['month'] = df_out['date_temp'].dt.month.astype('int8')
            df_out['quarter'] = df_out['date_temp'].dt.quarter.astype('int8')
            df_out['day_of_month'] = df_out['date_temp'].dt.day.astype('int8')
            try:
                 df_out['week_of_year'] = df_out['date_temp'].dt.isocalendar().week.astype('uint8')
            except AttributeError:
                 # للنسخ الأقدم من Pandas
                 self.logger.warning("dt.isocalendar() غير متوفر، استخدام dt.week كبديل (قديم).")
                 df_out['week_of_year'] = df_out['date_temp'].dt.week.astype('uint8')
            # استخدام .date لتحويل Timestamp إلى كائن date قبل البحث
            df_out['is_holiday'] = df_out['date_temp'].dt.date.apply(lambda x: 1 if x in self.saudi_holidays else 0).astype('int8')
            df_out['is_month_start'] = df_out['date_temp'].dt.is_month_start.astype('int8')
            df_out['is_month_end'] = df_out['date_temp'].dt.is_month_end.astype('int8')
            df_out['day_of_year'] = df_out['date_temp'].dt.dayofyear.astype('int16')
            df_out['year'] = df_out['date_temp'].dt.year.astype('int16') # إضافة السنة

            # إزالة العمود المؤقت
            df_out = df_out.drop(columns=['date_temp'])
            # self.logger.debug("تمت إضافة ميزات الوقت بنجاح.")
        except Exception as e:
            self.logger.error(f"خطأ أثناء إضافة ميزات الوقت: {e}", exc_info=True)
            # لا تحذف العمود المؤقت إذا فشلت العملية للسماح بالتصحيح
            # قد ترغب في إرجاع df الأصلي في حالة الفشل
            return df # إرجاع الأصلي بدون التعديلات إذا حدث خطأ
        return df_out


    def _fill_missing_dates(self, df):
        # استخدام الكود المصحح من الاستجابة السابقة للتعامل مع < 3 تواريخ
        if df.empty or 'sale_date' not in df.columns:
             self.logger.warning("DataFrame فارغ أو عمود 'sale_date' مفقود، لا يمكن ملء التواريخ المفقودة.")
             return df

        try:
            # التأكد من أن sale_date هو datetime object وليس date object فقط
            df['sale_date'] = pd.to_datetime(df['sale_date'])
        except Exception as e_date_conv:
            self.logger.error(f"خطأ تحويل sale_date إلى datetime في بداية _fill_missing_dates: {e_date_conv}")
            return df # إرجاع DF الأصلي إذا فشل التحويل

        unique_dates_count = df['sale_date'].nunique()
        self.logger.info(f"عدد التواريخ الفريدة قبل الملء: {unique_dates_count}")

        # التعامل مع حالة أقل من تاريخين
        if unique_dates_count < 2:
            self.logger.warning(f"تم العثور على {unique_dates_count} تواريخ فريدة فقط. لا يمكن ملء التواريخ المفقودة بشكل موثوق. سيتم إضافة ميزات الوقت فقط.")
            df_out = df.copy() # العمل على نسخة
            try:
                df_out = self._add_time_features(df_out)
            except Exception as e_time:
                 self.logger.error(f"خطأ عند محاولة إضافة ميزات الوقت لبيانات قليلة: {e_time}")
            # لا حاجة لـ reset_index إذا لم يتم تغيير الفهرس
            return df_out.sort_values('sale_date')

        # التعامل مع حالة تاريخين بالضبط
        elif unique_dates_count == 2:
            self.logger.warning("تم العثور على تاريخين فريدين فقط. سيتم فرض التردد اليومي 'D'.")
            df_out = df.copy().set_index('sale_date').sort_index()
            try:
                df_out = df_out.asfreq('D')
                self.logger.info("تم فرض التردد 'D' بنجاح على البيانات ذات التاريخين.")
            except Exception as e_asfreq_2days:
                self.logger.error(f"فشل فرض التردد 'D' على البيانات ذات التاريخين: {e_asfreq_2days}. سيتم إرجاع البيانات الأصلية مع ميزات الوقت.")
                df_out = df.copy() # العودة للبيانات الأصلية
                try:
                    df_out = self._add_time_features(df_out)
                except Exception as e_time_2:
                     self.logger.error(f"خطأ عند محاولة إضافة ميزات الوقت لبيانات اليومين بعد فشل asfreq: {e_time_2}")
                return df_out.sort_values('sale_date').reset_index(drop=True) # إعادة الفهرس هنا

        # التعامل مع حالة 3 تواريخ أو أكثر
        else: # unique_dates_count >= 3
            self.logger.info("تم العثور على 3 تواريخ فريدة أو أكثر، محاولة استنتاج وفرض التردد 'D'.")
            df_out = df.copy().set_index('sale_date').sort_index()
            try:
                inferred_freq = pd.infer_freq(df_out.index)
                if inferred_freq == 'D':
                    self.logger.info("تم استنتاج التردد اليومي 'D'. التأكد من ملء الفجوات.")
                    df_out = df_out.asfreq('D')
                else:
                    if inferred_freq:
                         self.logger.warning(f"التردد المستنتج ليس يوميًا ('{inferred_freq}'). سيتم فرض التردد 'D'.")
                    else:
                         self.logger.warning("لم يتم استنتاج تردد واضح. سيتم فرض التردد 'D'.")
                    df_out = df_out.asfreq('D')
            except ValueError as e_infer:
                 self.logger.error(f"خطأ أثناء استنتاج التردد (مع >=3 تواريخ): {e_infer}. محاولة فرض 'D' مباشرة.")
                 try:
                      df_out = df_out.asfreq('D')
                 except Exception as e_asfreq_direct:
                      self.logger.error(f"فشل فرض التردد 'D' بعد خطأ الاستنتاج: {e_asfreq_direct}")
                      df_out = df.copy() # العودة للأصل
                      try:
                           df_out = self._add_time_features(df_out)
                      except Exception as e_time_3:
                            self.logger.error(f"خطأ عند إضافة ميزات الوقت بعد فشل asfreq: {e_time_3}")
                      return df_out.sort_values('sale_date').reset_index(drop=True) # إعادة الفهرس

        # --- بقية الدالة تطبق على df_out ---
        sales_cols = ['daily_sales', 'transaction_count', 'total_items']
        nan_filled_count = 0
        for col in sales_cols:
            if col in df_out.columns:
                original_nan_count = df_out[col].isnull().sum()
                if original_nan_count > 0:
                    df_out[col] = df_out[col].fillna(0)
                    nan_filled_count += original_nan_count
        if nan_filled_count > 0:
             self.logger.info(f"تم ملء إجمالي {nan_filled_count} قيمة NaN في أعمدة المبيعات/الكميات بـ 0 للأيام الجديدة.")

        df_out = df_out.reset_index()
        df_out = df_out.rename(columns={'index': 'sale_date'})

        # إعادة حساب ميزات الوقت للجدول بأكمله (df_out)
        try:
             # التأكد من أن sale_date هو datetime قبل تمريره
             df_out['sale_date'] = pd.to_datetime(df_out['sale_date'])
             df_out = self._add_time_features(df_out)
             self.logger.info("اكتمل ملء التواريخ المفقودة وإعادة حساب ميزات الوقت.")
        except Exception as e_time_final:
            self.logger.error(f"خطأ أثناء إضافة ميزات الوقت النهائية بعد الملء: {e_time_final}")

        return df_out.sort_values('sale_date').reset_index(drop=True)


    def _add_lag_features(self, df):
        if df.empty or 'daily_sales' not in df.columns:
             # self.logger.debug("DataFrame فارغ أو يفتقد 'daily_sales' في _add_lag_features.")
             return df
        if len(df) < 2 : # لا يمكن حساب اللاج إذا كان هناك صف واحد أو أقل
             self.logger.warning(f"عدد الصفوف ({len(df)}) قليل جدًا لحساب ميزات اللاج.")
             return df

        target = 'daily_sales'
        lags = [1, 2, 3, 7, 14, 21, 28, 30] # يمكن تعديل هذه القائمة
        df_out = df.copy()
        # التأكد من الفرز قبل shift
        df_out = df_out.sort_values('sale_date')

        self.logger.info(f"إضافة ميزات اللاج لـ '{target}' للفترات: {lags}")
        for lag in lags:
            col_name = f'sales_lag_{lag}'
            # تأكد من أن lag أصغر من طول الـ DataFrame
            if lag < len(df_out):
                 df_out[col_name] = df_out[target].shift(lag)
                 # التحويل إلى float32 سيتم في _clean_data
            else:
                 self.logger.warning(f"لا يمكن إنشاء اللاج {lag} لأن طول البيانات ({len(df_out)}) أصغر. سيتم تخطي هذا اللاج.")

        return df_out


    def _clean_data(self, df):
        if df.empty:
             self.logger.warning("DataFrame فارغ، لا يمكن التنظيف.")
             return df

        self.logger.info(f"بدء تنظيف البيانات النهائية... الشكل قبل التنظيف: {df.shape}")
        df_out = df.copy()

        # 1. تحديد الأعمدة المطلوبة (تضمين الأعمدة الأساسية والميزات التي تم إنشاؤها)
        base_cols = ['sale_date', 'daily_sales', 'transaction_count', 'total_items']
        time_cols = ['day_of_week', 'is_weekend', 'month', 'quarter', 'day_of_month',
                     'week_of_year', 'is_holiday', 'is_month_start', 'is_month_end',
                     'day_of_year', 'year']
        lag_cols = [col for col in df_out.columns if col.startswith('sales_lag_')]
        required_columns = base_cols + time_cols + lag_cols

        # الاحتفاظ فقط بالأعمدة المطلوبة الموجودة فعلاً
        existing_required_columns = [col for col in required_columns if col in df_out.columns]
        original_cols = df_out.columns.tolist()
        df_out = df_out[existing_required_columns]
        removed_cols = set(original_cols) - set(existing_required_columns)
        if removed_cols:
            self.logger.info(f"تمت إزالة الأعمدة غير المطلوبة/المفقودة: {list(removed_cols)}")

        # 2. التعامل مع القيم المفقودة (NaN)
        rows_before_drop = len(df_out)
        # الهدف (daily_sales): يجب ألا يكون NaN. (تم ملؤه بـ 0 في _fill_missing_dates للأيام الجديدة)
        if 'daily_sales' in df_out.columns and df_out['daily_sales'].isnull().any():
             self.logger.warning("تم العثور على NaN في 'daily_sales' بشكل غير متوقع. سيتم حذف الصفوف.")
             df_out = df_out.dropna(subset=['daily_sales'])

        # ميزات اللاج: NaN طبيعي في البداية. يمكن تركها أو ملؤها. (سنتركها الآن)
        nan_in_lags = df_out[lag_cols].isnull().sum().sum()
        if nan_in_lags > 0:
             self.logger.info(f"تم العثور على {nan_in_lags} قيمة NaN في ميزات اللاج (طبيعي في بداية السلسلة).")

        # ميزات الوقت: يجب ألا تحتوي على NaN إذا تمت إضافتها بشكل صحيح.
        time_cols_check = [col for col in time_cols if col in df_out.columns]
        nan_in_time_features = df_out[time_cols_check].isnull().sum().sum()
        if nan_in_time_features > 0:
             nan_time_cols_list = df_out[time_cols_check].columns[df_out[time_cols_check].isnull().any()].tolist()
             self.logger.error(f"*** خطأ: تم العثور على {nan_in_time_features} قيمة NaN في ميزات الوقت بعد الإنشاء! الأعمدة: {nan_time_cols_list}. سيتم محاولة ملء بـ 0.")
             df_out[nan_time_cols_list] = df_out[nan_time_cols_list].fillna(0)

        if len(df_out) < rows_before_drop:
            self.logger.warning(f"تم حذف {rows_before_drop - len(df_out)} صفوف أثناء التنظيف (بسبب NaN في الهدف).")

        if df_out.empty:
             self.logger.error("أصبح DataFrame فارغًا بعد التنظيف.")
             return df_out # إرجاع فارغ

        # 3. تحويل أنواع البيانات لتوفير الذاكرة
        for col in df_out.select_dtypes(include=['float64']).columns:
            df_out[col] = df_out[col].astype('float32')
        for col in df_out.select_dtypes(include=['int64', 'Int64']).columns:
            # استثناء أعمدة ID المحتملة إذا كانت كبيرة جدًا
            if col not in ['id', 'invoice_id', 'product_id', 'user_id']:
                 try:
                     # التحويل إلى أصغر نوع ممكن
                     df_out[col] = pd.to_numeric(df_out[col], downcast='integer')
                 except Exception: # تجاهل الأخطاء المحتملة مع downcast وأنواع Int64
                      pass
        # تحويل الأعمدة المنطقية (0/1) إلى int8
        bool_like_cols = ['is_weekend', 'is_holiday', 'is_month_start', 'is_month_end']
        for col in bool_like_cols:
            if col in df_out.columns:
                 df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(0).astype('int8')

        # 4. التأكد من فرز البيانات وإعادة تعيين الفهرس
        df_out = df_out.sort_values('sale_date').reset_index(drop=True)

        self.logger.info(f"اكتمل تنظيف البيانات. الشكل النهائي: {df_out.shape}")
        # self.logger.debug(f"أنواع البيانات النهائية: \n{df_out.dtypes}")
        return df_out

    def generate_features(self):
        """
        الدالة الرئيسية لتشغيل خطوات هندسة الميزات بالترتيب.
        """
        try:
            self.logger.info("--- بدء عملية إنشاء الميزات ---")
            self._preprocess_dates()
            daily_sales = self._create_daily_sales()

            if daily_sales.empty:
                 # _create_daily_sales يجب أن تكون قد سجلت السبب
                 self.logger.error("فشلت خطوة إنشاء المبيعات اليومية أو لم تنتج بيانات.")
                 # يمكن إرجاع DataFrame فارغ هنا إذا أردنا السماح للخطوات التالية بالمحاولة
                 # أو رفع الخطأ لإيقاف العملية مبكرًا
                 raise ValueError("فشل إنشاء بيانات المبيعات اليومية الأولية.")

            # إضافة ميزات الوقت الأساسية
            daily_sales = self._add_time_features(daily_sales)
            # ملء التواريخ المفقودة وإعادة حساب ميزات الوقت
            daily_sales_filled = self._fill_missing_dates(daily_sales)
            # إضافة ميزات اللاج
            daily_sales_lagged = self._add_lag_features(daily_sales_filled)
            # تنظيف نهائي وتحديد الأنواع
            self.features = self._clean_data(daily_sales_lagged)

            self.logger.info("--- اكتملت عملية إنشاء الميزات بنجاح ---")
            return self.features

        except (FileNotFoundError, ValueError, KeyError) as data_err:
             self.logger.error(f"فشل في إنشاء الميزات بسبب خطأ في البيانات أو الملفات: {str(data_err)}", exc_info=True)
             raise # إعادة رفع الخطأ ليتم التعامل معه في الكود المستدعي
        except Exception as e:
            self.logger.error(f"فشل عام غير متوقع في إنشاء الميزات: {str(e)}", exc_info=True)
            raise RuntimeError(f"فشل عام في إنشاء الميزات: {e}")


# --- **** الدالة العامة القابلة للاستدعاء **** ---
def generate_features_df(data_paths):
    """
    تقوم بتحميل بيانات المبيعات، إجراء هندسة الميزات، وإرجاع DataFrame جاهز للتنبؤ.

    Args:
        data_paths (dict): قاموس يحتوي على مسارات ملفات البيانات المطلوبة.
                           مثال: {'sale_invoices': 'path/invoices.xlsx',
                                  'sale_invoices_details': 'path/details.xlsx'}

    Returns:
        pandas.DataFrame: DataFrame يحتوي على المبيعات اليومية والميزات المشتقة،
                          أو يثير استثناء (Exception) في حالة الفشل.
    """
    logger.info("--- بدء دالة generate_features_df ---")
    try:
        engineer = _SalesFeatureEngineerInternal(data_paths)
        features_df = engineer.generate_features()

        if features_df is None or features_df.empty:
             logger.error("فشلت عملية إنشاء الميزات وأرجعت DataFrame فارغًا أو None.")
             raise ValueError("فشل إنشاء الميزات (النتيجة فارغة).")

        required_output_cols = ['sale_date', 'daily_sales']
        if not all(col in features_df.columns for col in required_output_cols):
            missing_cols = list(set(required_output_cols) - set(features_df.columns))
            logger.error(f"الـ DataFrame الناتج يفتقد لأعمدة أساسية: {missing_cols}")
            raise ValueError(f"الـ DataFrame الناتج غير مكتمل (مفقود: {missing_cols}).")

        if 'sale_date' in features_df.columns and features_df['sale_date'].isnull().any():
            logger.error("الـ DataFrame الناتج يحتوي على قيم NaN في عمود 'sale_date'.")
            raise ValueError("قيم تاريخ مفقودة في النتيجة النهائية للميزات.")

        logger.info(f"--- اكتملت دالة generate_features_df بنجاح. الشكل: {features_df.shape} ---")
        return features_df

    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as data_err:
        logger.error(f"خطأ مُعالج في generate_features_df: {data_err}")
        raise # إعادة رفع الخطأ المحدد
    except Exception as e:
        logger.error(f"خطأ عام غير متوقع في generate_features_df: {e}", exc_info=True)
        raise RuntimeError(f"فشل عام في إنشاء الميزات: {e}")


# --- مثال للاختبار المستقل (اختياري) ---
if __name__ == "__main__":
    print("--- اختبار مستقل لـ feature_engineering.py ---")
    # !!! قم بتحديث هذه المسارات لتناسب بيئتك !!!
    test_data_paths = {
        'sale_invoices': r'C:\Users\sheee\Downloads\ZodData\sale_invoices.xlsx',
        'sale_invoices_details': r'C:\Users\sheee\Downloads\ZodData\sale_invoices_details.xlsx'
    }
    try:
        print(f"استدعاء generate_features_df بالمسارات: {test_data_paths}")
        features_result_df = generate_features_df(test_data_paths)
        print("\n--- نجح إنشاء الميزات (اختبار مستقل) ---")
        print(f"شكل DataFrame الناتج: {features_result_df.shape}")
        if not features_result_df.empty:
            print(f"الفترة الزمنية: من {features_result_df['sale_date'].min().date()} إلى {features_result_df['sale_date'].max().date()}")
            print("\nأول 5 صفوف:")
            print(features_result_df.head())
            print("\nآخر 5 صفوف:")
            print(features_result_df.tail())
            print("\nمعلومات DataFrame:")
            features_result_df.info()

            # التحقق من عدم وجود أيام ناقصة ضمن النطاق الزمني
            date_range_check = pd.date_range(start=features_result_df['sale_date'].min(), end=features_result_df['sale_date'].max())
            missing_dates_check = date_range_check[~date_range_check.isin(features_result_df['sale_date'])]
            if not missing_dates_check.empty:
                 print(f"\n*** تحذير: تم العثور على {len(missing_dates_check)} يومًا ناقصًا ضمن النطاق الزمني بعد المعالجة!")
                 # print(missing_dates_check)
            else:
                 print("\nالتحقق من التواريخ: لا توجد أيام ناقصة ضمن النطاق الزمني للبيانات المعالجة.")
        else:
             print("DataFrame الناتج فارغ.")

    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"\n--- فشل الاختبار المستقل بخطأ متوقع: {str(e)} ---")
    except Exception as e_main:
         print(f"\n--- فشل الاختبار المستقل بخطأ غير متوقع: {str(e_main)} ---")
         import traceback
         traceback.print_exc()