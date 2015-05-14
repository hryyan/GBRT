#include "tools.h"

pair<Mat, Mat> read_data_from_txt(QString str)
{
    QFile f(str);
    QList<QVector<double>> feats;
    QList<int> target;
    if (f.open(QIODevice::ReadOnly))
    {
        while (!f.atEnd())
        {
            QByteArray line = f.readLine();
            line.resize(line.size()-1);
            QList<QByteArray> l = line.split(' ');

            target.push_back(l.at(0).toInt());
            QVector<double> feat;
            for (int i = 1; i < l.size(); i++)
            {
                feat.push_back(l.at(i).toDouble());
            }
            feats.push_back(feat);
        }
    }

    int n_feature = feats[0].size();
    int n_sample = feats.size();

    Mat m_feat(n_sample, n_feature, CV_64F);
    Mat m_target(n_sample, 1, CV_64F);

    for (int i = 0; i < feats.size(); i++)
    {
        for (int j = 0; j < feats.at(i).size(); j++)
        {
            m_feat.at<double>(i, j) = feats.at(i).at(j);
        }
        m_target.at<double>(i) = target.at(i);
    }
    return make_pair(m_feat, m_target);
}

pair<Mat, Mat> read_data_from_txt_regression(QString str)
{
    QFile f(str);
    QList<QVector<double>> feats;
    QList<double> target;
    if (f.open(QIODevice::ReadOnly))
    {
        while (!f.atEnd())
        {
            QByteArray line = f.readLine();
            line.resize(line.size()-1);
            QList<QByteArray> l = line.split(' ');

            target.push_back(l.at(0).toDouble());
            QVector<double> feat;
            for (int i = 1; i < l.size(); i++)
            {
                feat.push_back(l.at(i).toDouble());
            }
            feats.push_back(feat);
        }
    }

    int n_feature = feats[0].size();
    int n_sample = feats.size();

    Mat m_feat(n_sample, n_feature, CV_64F);
    Mat m_target(n_sample, 1, CV_64F);

    for (int i = 0; i < feats.size(); i++)
    {
        for (int j = 0; j < feats.at(i).size(); j++)
        {
            m_feat.at<double>(i, j) = feats.at(i).at(j);
        }
        m_target.at<double>(i) = target.at(i);
    }
    return make_pair(m_feat, m_target);
}
