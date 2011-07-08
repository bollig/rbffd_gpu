function [] = plot_eigenvalues(filename)

evals = dlmread(filename);
plot(real(evals), imag(evals), 'o','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','g','MarkerSize',8);
axis tight;
grid on;
title(filename,'Interpreter', 'None', 'FontSize',24);
xlabel('Real', 'FontSize',24);
ylabel('Imag', 'FontSize',24);
end