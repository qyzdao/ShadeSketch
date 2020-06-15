const path = require('path');
const webpack = require('webpack');
const VueLoaderPlugin = require('vue-loader/lib/plugin');
const UglifyJsPlugin = require("uglifyjs-webpack-plugin");
const htmlWebpackPlugin = require('html-webpack-plugin');

const config = {
	entry: path.join(__dirname, 'src/index.js'),
	output: {
		path: path.resolve(__dirname, 'dist'),
		filename: 'bundle.js'
	},
	devServer: {
		contentBase: path.join(__dirname),
	},
	resolve: {
		extensions: ['.js', '.vue'],
		alias: {
			vue: 'vue/dist/vue.js',
		}
	},
	devtool: 'cheap-module-source-map',
	module: {
		rules: [{
				test: /\.vue$/,
				loader: 'vue-loader',
				exclude: /node_modules/,
				include: path.resolve(__dirname, 'src')
			}, {
				test: /\.js$/,
				exclude: /node_modules/,
				loader: 'babel-loader',
				include: path.resolve(__dirname, 'src')
			},
			{
				test: /\.css$/,
				use: [
					'vue-style-loader',
					'css-loader'
				]
			}
		]
	},
	plugins: [
		new VueLoaderPlugin(),
		new htmlWebpackPlugin({
			template: path.join(__dirname, './src/index.html'),
			filename: 'index.html'
		})
	]
};

if (process.env.NODE_ENV === 'production') {
	config.plugins = [new webpack.DefinePlugin({
		'process.env.NODE_ENV': JSON.stringify('production')
	})];
}

module.exports = config;