import React from 'react';
import { Card, Button, Typography, Row, Col, Space, Divider } from 'antd';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  CameraOutlined,
  EditOutlined,
  DownloadOutlined,
  StarOutlined,
  RocketOutlined,
  UserOutlined
} from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <CameraOutlined style={{ fontSize: '48px', color: '#1890ff' }} />,
      title: '智能拍照上传',
      description: '支持多种图片格式，自动检测人脸，确保最佳处理效果'
    },
    {
      icon: <EditOutlined style={{ fontSize: '48px', color: '#52c41a' }} />,
      title: '文本描述发型',
      description: '通过自然语言描述您想要的发型，AI智能理解并生成'
    },
    {
      icon: <DownloadOutlined style={{ fontSize: '48px', color: '#fa8c16' }} />,
      title: '高清结果下载',
      description: '生成高质量图片，支持多种格式下载，保存您的完美形象'
    }
  ];

  const examples = [
    {
      before: '/images/example1-before.jpg',
      after: '/images/example1-after.jpg',
      title: '短发变长发',
      rating: 5,
      comment: '效果超级自然！朋友都认不出我了'
    },
    {
      before: '/images/example2-before.jpg',
      after: '/images/example2-after.jpg',
      title: '直发变卷发',
      rating: 5,
      comment: '终于找到适合我的发型了'
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Hero Section */}
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="container mx-auto px-4 py-16"
      >
        <div className="text-center mb-16">
          <motion.div variants={itemVariants}>
            <Title level={1} className="text-5xl font-bold mb-6 gradient-text">
              ✨ AI智能换发型 ✨
            </Title>
          </motion.div>

          <motion.div variants={itemVariants}>
            <Paragraph className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              上传您的照片，瞬间变换理想发型
              <br />
              基于先进的AI技术，保持面部特征完全不变
            </Paragraph>
          </motion.div>

          <motion.div variants={itemVariants} className="flex justify-center gap-4 mb-12">
            <Button
              type="primary"
              size="large"
              icon={<CameraOutlined />}
              onClick={() => navigate('/upload')}
              className="btn-gradient text-lg px-8 py-4 h-auto"
            >
              🎯 文本描述换发型
            </Button>
            <Button
              size="large"
              icon={<EditOutlined />}
              onClick={() => navigate('/upload')}
              className="text-lg px-8 py-4 h-auto"
            >
              📸 参考图片换发型
            </Button>
          </motion.div>
        </div>

        {/* Features Section */}
        <motion.div variants={itemVariants} className="mb-16">
          <Title level={2} className="text-center mb-12">
            核心功能
          </Title>
          <Row gutter={[24, 24]}>
            {features.map((feature, index) => (
              <Col xs={24} md={8} key={index}>
                <motion.div variants={itemVariants}>
                  <Card
                    hoverable
                    className="feature-card h-full text-center"
                    bodyStyle={{ padding: '32px' }}
                  >
                    <div className="mb-4">{feature.icon}</div>
                    <Title level={4} className="mb-3">
                      {feature.title}
                    </Title>
                    <Paragraph className="text-gray-600">
                      {feature.description}
                    </Paragraph>
                  </Card>
                </motion.div>
              </Col>
            ))}
          </Row>
        </motion.div>

        {/* Examples Section */}
        <motion.div variants={itemVariants}>
          <Title level={2} className="text-center mb-12">
            用户案例
          </Title>
          <Row gutter={[24, 24]} justify="center">
            {examples.map((example, index) => (
              <Col xs={24} md={10} key={index}>
                <motion.div variants={itemVariants}>
                  <Card
                    hoverable
                    className="feature-card"
                    cover={
                      <div className="relative">
                        <img
                          src={example.before}
                          alt={`${example.title} - 原图`}
                          className="w-full h-48 object-cover"
                        />
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-black/30 flex items-end p-4">
                          <span className="text-white font-semibold">原图</span>
                        </div>
                      </div>
                    }
                  >
                    <div className="text-center mb-4">
                      <Title level={5} className="mb-2">
                        {example.title}
                      </Title>
                      <div className="flex justify-center mb-2">
                        {[...Array(example.rating)].map((_, i) => (
                          <StarOutlined key={i} className="text-yellow-400" />
                        ))}
                      </div>
                      <Paragraph className="text-gray-600 italic">
                        "{example.comment}"
                      </Paragraph>
                    </div>
                  </Card>
                </motion.div>
              </Col>
            ))}
          </Row>
        </motion.div>

        {/* CTA Section */}
        <motion.div variants={itemVariants} className="text-center mt-16">
          <Divider />
          <Title level={3} className="mb-4">
            准备好变换您的发型了吗？
          </Title>
          <Paragraph className="text-lg text-gray-600 mb-8">
            现在就开始体验AI换发型的神奇效果
          </Paragraph>
          <Button
            type="primary"
            size="large"
            icon={<RocketOutlined />}
            onClick={() => navigate('/upload')}
            className="btn-gradient text-lg px-12 py-4 h-auto"
          >
            开始体验 →
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default HomePage;