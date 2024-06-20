
import pandas as pd
import pymysql
import pymysql.cursors

from utils import main


if __name__ == "__main__":
    """
    """
    df = pd.read_csv('666.csv')
    with open('data_cleaning/train.txt', 'a+', encoding='utf-8') as file:
        for i in range(len(df)-3030):
            i = i+3030
            cn = df['company_name'][i] + ','
            ca = df['company_area'][i] + ','
            jn = df['job_name'][i] + ','
            detail = df['detail'][i]
            input = detail
            output = cn + ca + jn + main(input)
            file.write(output+'\n')
            print(f"写入成功！第{i+1}条数据")
            print(f'内容为: ', output)